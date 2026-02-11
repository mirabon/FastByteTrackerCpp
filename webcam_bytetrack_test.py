# file: webcam_bytetrack_test.py
"""
Webcam BYTETrack smoke-test (no neural detector).
Detections are produced by motion segmentation (MOG2) -> bounding boxes.
Install deps:
  python -m pip install opencv-python
Run:
  python webcam_bytetrack_test.py --camera 0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2

try:
    from bytetrack_cpp import Tracker
except Exception as e:
    print("Failed to import bytetrack_cpp. Did you install it into this venv?", file=sys.stderr)
    raise


Det = Tuple[float, float, float, float, float]  # (x, y, w, h, score)


@dataclass(frozen=True)
class Config:
    camera: int
    width: int
    height: int
    fps: int
    track_buffer: int
    min_area: int
    show_mask: bool
    max_boxes: int


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0, help="Webcam index (usually 0)")
    p.add_argument("--width", type=int, default=1280, help="Capture width (0=keep default)")
    p.add_argument("--height", type=int, default=720, help="Capture height (0=keep default)")
    p.add_argument("--fps", type=int, default=30, help="FPS passed to tracker (best-effort)")
    p.add_argument("--track-buffer", type=int, default=30, help="BYTETrack track_buffer")
    p.add_argument("--min-area", type=int, default=900, help="Min contour area to keep bbox")
    p.add_argument("--max-boxes", type=int, default=50, help="Cap detections per frame")
    p.add_argument("--show-mask", action="store_true", help="Show motion mask window")
    a = p.parse_args()

    return Config(
        camera=a.camera,
        width=a.width,
        height=a.height,
        fps=a.fps,
        track_buffer=a.track_buffer,
        min_area=a.min_area,
        show_mask=a.show_mask,
        max_boxes=a.max_boxes,
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


def motion_detections(
    frame_bgr,
    subtractor,
    min_area: int,
    max_boxes: int,
) -> Tuple[List[Det], "cv2.Mat"]:
    mask = subtractor.apply(frame_bgr)

    # shadows -> drop; keep only strong foreground
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets: List[Det] = []
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)

    # Sort large-to-small to keep most confident first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:max_boxes]:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Heuristic confidence: bigger moving blob -> higher score
        score = min(0.99, max(0.10, area / max(frame_area * 0.05, 1.0)))
        dets.append((float(x), float(y), float(bw), float(bh), float(score)))

    return dets, mask


def draw_tracks(frame_bgr, tracks: Sequence[dict]) -> None:
    for tr in tracks:
        tlwh = tr.get("tlwh")
        if not tlwh or len(tlwh) != 4:
            continue

        x, y, w, h = tlwh
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        track_id = int(tr.get("track_id", -1))
        label = int(tr.get("label", 0))
        score = float(tr.get("score", 0.0))

        # Deterministic pseudo-color by id (BGR)
        c = (
            (track_id * 37) % 255,
            (track_id * 17) % 255,
            (track_id * 97) % 255,
        )

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, 2)
        text = f"id={track_id} cls={label} s={score:.2f}"
        cv2.putText(frame_bgr, text, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)


def main() -> int:
    cfg = parse_args()
    cap = open_camera(cfg)

    tracker = Tracker(frame_rate=cfg.fps, track_buffer=cfg.track_buffer)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True,
    )

    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame from camera", file=sys.stderr)
                break

            dets, mask = motion_detections(
                frame_bgr=frame,
                subtractor=subtractor,
                min_area=cfg.min_area,
                max_boxes=cfg.max_boxes,
            )

            tracks = tracker.update(dets)
            draw_tracks(frame, tracks)

            cv2.imshow("BYTETrack webcam test", frame)
            if cfg.show_mask:
                cv2.imshow("motion mask", mask)

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
