# file: webcam_yolo11_bytetrack.py
"""
YOLO11 detections -> bytetrack_cpp Tracker (webcam).

Run:
  python webcam_yolo11_bytetrack.py --model /path/to/yolo11.pt --camera 0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from bytetrack_cpp import Tracker as ByteTracker

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Install: python -m pip install ultralytics") from e


Det = Tuple[float, float, float, float, int, float]  # x,y,w,h,label,score


@dataclass(frozen=True)
class Config:
    model_path: str
    camera: int
    imgsz: int
    conf: float
    iou: float
    device: str
    max_det: int
    classes: Optional[List[int]]
    width: int
    height: int
    fps: int
    track_buffer: int
    show_dets: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model_path", required=True, help="Path to your YOLO11 .pt weights")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default="", help="''=auto, 'cpu', '0' etc.")
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--classes", type=str, default="", help="Comma-separated class ids, e.g. '0,2,3'")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30, help="Passed into BYTETrack")
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--show-dets", action="store_true", help="Also draw raw YOLO detections")
    a = p.parse_args()

    classes = None
    if a.classes.strip():
        classes = [int(x) for x in a.classes.split(",") if x.strip()]

    return Config(
        model_path=a.model_path,
        camera=a.camera,
        imgsz=a.imgsz,
        conf=float(a.conf),
        iou=float(a.iou),
        device=str(a.device),
        max_det=int(a.max_det),
        classes=classes,
        width=int(a.width),
        height=int(a.height),
        fps=int(a.fps),
        track_buffer=int(a.track_buffer),
        show_dets=bool(a.show_dets),
    )


def open_camera(cfg: Config) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index={cfg.camera}")

    if cfg.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
    if cfg.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
    if cfg.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(cfg.fps))

    return cap


def id_color(track_id: int) -> Tuple[int, int, int]:
    return (int((track_id * 37) % 255), int((track_id * 17) % 255), int((track_id * 97) % 255))


def yolo_to_dets(result, max_det: int) -> List[Det]:
    """
    Convert Ultralytics Results -> list of (x,y,w,h,label,score) in pixel coords.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls

    xyxy = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
    conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
    cls = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)

    n = min(len(xyxy), max_det)
    dets: List[Det] = []
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i].tolist()
        score = float(conf[i])
        label = int(cls[i])
        x = float(x1)
        y = float(y1)
        w = float(max(0.0, x2 - x1))
        h = float(max(0.0, y2 - y1))
        if w <= 1.0 or h <= 1.0:
            continue
        dets.append((x, y, w, h, label, score))
    return dets


def draw_dets(frame, dets: Sequence[Det], names: Optional[dict] = None) -> None:
    for x, y, w, h, label, score in dets:
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cls_name = str(label) if not names else names.get(label, str(label))
        cv2.putText(
            frame,
            f"{cls_name}:{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            2,
        )


def draw_tracks(frame, tracks: Sequence[dict], names: Optional[dict] = None) -> None:
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
        cls_name = str(label) if not names else names.get(label, str(label))
        cv2.putText(
            frame,
            f"id={tid} {cls_name} s={score:.2f}",
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            c,
            2,
        )


def main() -> int:
    cfg = parse_args()

    model = YOLO(cfg.model_path)
    names = getattr(model, "names", None)

    cap = open_camera(cfg)
    bytetrack = ByteTracker(frame_rate=cfg.fps, track_buffer=cfg.track_buffer)

    paused = False
    last_t = time.time()
    fps_smooth: Optional[float] = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Ultralytics expects BGR OK; it will handle conversion internally as needed.
            results = model.predict(
                source=frame,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=cfg.iou,
                device=cfg.device,
                classes=cfg.classes,
                max_det=cfg.max_det,
                verbose=False,
            )
            res0 = results[0]
            dets = yolo_to_dets(res0, cfg.max_det)

            # Feed to BYTETrack. Your binding should accept (x,y,w,h,label,score).
            tracks = bytetrack.update(dets)

            if cfg.show_dets:
                draw_dets(frame, dets, names=names)
            draw_tracks(frame, tracks, names=names)

            now = time.time()
            dt = max(now - last_t, 1e-6)
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
            last_t = now

            cv2.putText(
                frame,
                f"YOLO dets: {len(dets)} | BYTE tracks: {len(tracks)} | FPS: {fps_smooth:.1f}",
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

            cv2.imshow("YOLO11 -> BYTETrack (webcam)", frame)

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
