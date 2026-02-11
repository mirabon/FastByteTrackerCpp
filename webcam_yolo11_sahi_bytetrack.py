# file: webcam_yolo11_sahi_bytetrack.py
"""
YOLO11 (Ultralytics) + SAHI sliced inference -> bytetrack_cpp Tracker.

Install:
  python -m pip install -U ultralytics sahi opencv-python

Run (webcam 0):
  python webcam_yolo11_sahi_bytetrack.py --model /path/to/yolo11.pt --source 0

Run (video file):
  python webcam_yolo11_sahi_bytetrack.py --model /path/to/yolo11.pt --source /path/to/video.mp4

Keys:
  p       pause/resume
  q/ESC   quit
"""

from __future__ import annotations

import argparse
import inspect
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2

from bytetrack_cpp import Tracker as ByteTracker
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


Det = Tuple[float, float, float, float, int, float]  # x, y, w, h, label, score


@dataclass(frozen=True)
class Config:
    model_path: str
    source: str
    device: str
    imgsz: int
    conf: float

    slice_height: int
    slice_width: int
    overlap_h: float
    overlap_w: float

    postprocess_type: Optional[str]
    postprocess_match_threshold: Optional[float]
    postprocess_class_agnostic: Optional[bool]

    classes: Optional[List[int]]
    max_det: int

    width: int
    height: int
    fps: int

    track_buffer: int
    detect_every: int
    show_dets: bool


def _filter_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    supported = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in supported and v is not None}


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model_path", required=True, help="Path to your YOLO11 .pt weights")
    p.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or video path")
    p.add_argument("--device", type=str, default="", help="''=auto, 'cpu', 'cuda:0', etc.")
    p.add_argument("--imgsz", type=int, default=640, help="SAHI model image_size (Ultralytics inference size)")
    p.add_argument("--conf", type=float, default=0.35, help="SAHI confidence_threshold")

    p.add_argument("--slice-h", type=int, default=640, help="SAHI slice_height")
    p.add_argument("--slice-w", type=int, default=640, help="SAHI slice_width")
    p.add_argument("--overlap-h", type=float, default=0.2, help="SAHI overlap_height_ratio")
    p.add_argument("--overlap-w", type=float, default=0.2, help="SAHI overlap_width_ratio")

    p.add_argument("--postprocess-type", type=str, default="NMS", help="SAHI postprocess_type (e.g. NMS)")
    p.add_argument("--postprocess-match-thr", type=float, default=0.5, help="SAHI postprocess_match_threshold")
    p.add_argument(
        "--postprocess-class-agnostic",
        action="store_true",
        help="SAHI postprocess_class_agnostic (if supported by your SAHI version)",
    )

    p.add_argument("--classes", type=str, default="", help="Comma-separated class ids, e.g. '0,2,3'")
    p.add_argument("--max-det", type=int, default=200, help="Cap SAHI detections per frame")

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30, help="Passed into BYTETrack")

    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--detect-every", type=int, default=1, help="Run SAHI every N frames (others: update([]))")
    p.add_argument("--show-dets", action="store_true", help="Draw raw SAHI detections too")
    a = p.parse_args()

    classes = None
    if a.classes.strip():
        classes = [int(x) for x in a.classes.split(",") if x.strip()]

    post_type = a.postprocess_type.strip() if a.postprocess_type else None
    if post_type == "":
        post_type = None

    return Config(
        model_path=a.model_path,
        source=a.source,
        device=a.device,
        imgsz=int(a.imgsz),
        conf=float(a.conf),
        slice_height=int(a.slice_h),
        slice_width=int(a.slice_w),
        overlap_h=float(a.overlap_h),
        overlap_w=float(a.overlap_w),
        postprocess_type=post_type,
        postprocess_match_threshold=float(a.postprocess_match_thr),
        postprocess_class_agnostic=bool(a.postprocess_class_agnostic),
        classes=classes,
        max_det=int(a.max_det),
        width=int(a.width),
        height=int(a.height),
        fps=int(a.fps),
        track_buffer=int(a.track_buffer),
        detect_every=max(1, int(a.detect_every)),
        show_dets=bool(a.show_dets),
    )


def open_capture(cfg: Config) -> cv2.VideoCapture:
    src: Union[int, str]
    src = int(cfg.source) if cfg.source.isdigit() else cfg.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source={cfg.source}")

    if cfg.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
    if cfg.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
    if cfg.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(cfg.fps))
    return cap


def create_sahi_model(cfg: Config):
    kwargs = dict(
        model_type="ultralytics",
        model_path=cfg.model_path,
        confidence_threshold=cfg.conf,
        device=cfg.device,
        image_size=cfg.imgsz,
    )
    return AutoDetectionModel.from_pretrained(**_filter_supported_kwargs(AutoDetectionModel.from_pretrained, kwargs))


def sahi_result_to_dets(
    result: Any,
    max_det: int,
    allowed_classes: Optional[Sequence[int]],
) -> List[Det]:
    out: List[Det] = []
    opl = getattr(result, "object_prediction_list", None)
    if not opl:
        return out

    for op in opl:
        bbox = getattr(op, "bbox", None)
        cat = getattr(op, "category", None)
        score = getattr(op, "score", None)
        if bbox is None or cat is None or score is None:
            continue

        label = int(getattr(cat, "id", -1))
        if allowed_classes is not None and label not in allowed_classes:
            continue

        minx = float(getattr(bbox, "minx", 0.0))
        miny = float(getattr(bbox, "miny", 0.0))
        maxx = float(getattr(bbox, "maxx", 0.0))
        maxy = float(getattr(bbox, "maxy", 0.0))
        conf = float(getattr(score, "value", 0.0))

        w = max(0.0, maxx - minx)
        h = max(0.0, maxy - miny)
        if w <= 1.0 or h <= 1.0:
            continue

        out.append((minx, miny, w, h, label, conf))
        if len(out) >= max_det:
            break

    return out


def id_color(track_id: int) -> Tuple[int, int, int]:
    return (int((track_id * 37) % 255), int((track_id * 17) % 255), int((track_id * 97) % 255))


def draw_dets(frame, dets: Sequence[Det]) -> None:
    for x, y, w, h, label, score in dets:
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(
            frame,
            f"cls={label} s={score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            2,
        )


def draw_tracks(frame, tracks: Sequence[dict]) -> None:
    for tr in tracks:
        tlwh = tr.get("tlwh")
        if not tlwh or len(tlwh) != 4:
            continue
        x, y, w, h = tlwh
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        tid = int(tr.get("track_id", -1))
        label = int(tr.get("label", -1))
        score = float(tr.get("score", 0.0))

        c = id_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            frame,
            f"id={tid} cls={label} s={score:.2f}",
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            c,
            2,
        )


def main() -> int:
    cfg = parse_args()

    cap = open_capture(cfg)
    sahi_model = create_sahi_model(cfg)
    bytetrack = ByteTracker(frame_rate=cfg.fps, track_buffer=cfg.track_buffer)

    paused = False
    frame_idx = 0

    last_t = time.time()
    fps_smooth: Optional[float] = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            dets: List[Det] = []
            if frame_idx % cfg.detect_every == 0:
                pred_kwargs = dict(
                    slice_height=cfg.slice_height,
                    slice_width=cfg.slice_width,
                    overlap_height_ratio=cfg.overlap_h,
                    overlap_width_ratio=cfg.overlap_w,
                    postprocess_type=cfg.postprocess_type,
                    postprocess_match_threshold=cfg.postprocess_match_threshold,
                    postprocess_class_agnostic=cfg.postprocess_class_agnostic,
                    verbose=0,
                )
                pred_kwargs = _filter_supported_kwargs(get_sliced_prediction, pred_kwargs)

                result = get_sliced_prediction(frame, sahi_model, **pred_kwargs)
                dets = sahi_result_to_dets(result, max_det=cfg.max_det, allowed_classes=cfg.classes)
                # tracks = bytetrack.update(dets)
                tracks = bytetrack.update([])
            else:
                tracks = bytetrack.update([])

            if cfg.show_dets and dets:
                draw_dets(frame, dets)
            draw_tracks(frame, tracks)

            now = time.time()
            dt = max(now - last_t, 1e-6)
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
            last_t = now

            cv2.putText(
                frame,
                f"SAHI dets: {len(dets)} | BYTE tracks: {len(tracks)} | FPS: {fps_smooth:.1f} | every={cfg.detect_every}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Keys: p=pause  q/ESC=quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("YOLO11 + SAHI -> BYTETrack", frame)
            frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
