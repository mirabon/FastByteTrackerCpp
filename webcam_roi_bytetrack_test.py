# file: webcam_roi_bytetrack_test.py
"""
Manual ROI selection (OpenCV-style) + feed boxes into bytetrack_cpp.

Deps:
  python -m pip install opencv-contrib-python
Run:
  python webcam_roi_bytetrack_test.py --camera 0 --tracker csrt

Keys:
  s  - select ROI(s) again
  c  - clear all ROI trackers
  p  - pause/resume
  q/ESC - quit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2

from bytetrack_cpp import Tracker as ByteTracker

ROI = Tuple[int, int, int, int]          # x, y, w, h
Det = Tuple[float, float, float, float, float]  # x, y, w, h, score


@dataclass(frozen=True)
class Config:
    camera: int
    width: int
    height: int
    fps: int
    track_buffer: int
    tracker_name: str
    multi_select: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30, help="Value passed to BYTETrack")
    p.add_argument("--track-buffer", type=int, default=6)
    p.add_argument(
        "--tracker",
        dest="tracker_name",
        type=str,
        default="csrt",
        choices=["csrt", "kcf", "mosse"],
        help="OpenCV single-object tracker used to generate boxes",
    )
    p.add_argument(
        "--multi-select",
        action="store_true",
        help="Select multiple ROIs (if OpenCV supports selectROIs)",
    )
    a = p.parse_args()
    return Config(
        camera=a.camera,
        width=a.width,
        height=a.height,
        fps=a.fps,
        track_buffer=a.track_buffer,
        tracker_name=a.tracker_name.lower(),
        multi_select=bool(a.multi_select),
    )


def create_opencv_tracker(name: str):
    """
    OpenCV tracker factory with compatibility for cv2.legacy.
    Requires opencv-contrib-python for most trackers.
    """
    legacy = getattr(cv2, "legacy", None)

    def _make(mod, fname: str):
        fn = getattr(mod, fname, None)
        return fn() if fn else None

    if name == "csrt":
        return _make(legacy, "TrackerCSRT_create") or _make(cv2, "TrackerCSRT_create")
    if name == "kcf":
        return _make(legacy, "TrackerKCF_create") or _make(cv2, "TrackerKCF_create")
    if name == "mosse":
        return _make(legacy, "TrackerMOSSE_create") or _make(cv2, "TrackerMOSSE_create")

    raise ValueError(f"Unknown tracker: {name}")


def select_rois(frame, multi: bool) -> List[ROI]:
    """
    Returns list of ROIs. Empty list if user cancels.
    """
    win = "Select ROI (ENTER/SPACE to confirm, ESC to cancel)"
    cv2.imshow(win, frame)
    cv2.waitKey(1)

    rois: List[ROI] = []
    if multi and hasattr(cv2, "selectROIs"):
        arr = cv2.selectROIs(win, frame, showCrosshair=True, fromCenter=False)
        for x, y, w, h in arr:
            if w > 0 and h > 0:
                rois.append((int(x), int(y)), int(w), int(h))  # type: ignore
        # Fix for the line above (keep it simple & correct):
        rois = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in arr if w > 0 and h > 0]
    else:
        x, y, w, h = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
        if w > 0 and h > 0:
            rois = [(int(x), int(y), int(w), int(h))]

    cv2.destroyWindow(win)
    return rois


def init_roi_trackers(frame, rois: Sequence[ROI], tracker_name: str):
    trackers = []
    for roi in rois:
        tr = create_opencv_tracker(tracker_name)
        if tr is None:
            raise RuntimeError(
                f"Failed to create OpenCV tracker '{tracker_name}'. "
                "Install opencv-contrib-python."
            )
        ok = tr.init(frame, roi)
        if ok:
            trackers.append(tr)
    return trackers


def update_roi_trackers(trackers, frame) -> List[ROI]:
    rois: List[ROI] = []
    alive = []
    for tr in trackers:
        ok, bbox = tr.update(frame)
        if not ok:
            continue
        x, y, w, h = bbox
        if w <= 1 or h <= 1:
            continue
        rois.append((int(x), int(y), int(w), int(h)))
        alive.append(tr)
    trackers[:] = alive
    return rois


def draw_byte_tracks(frame, tracks: Sequence[dict]) -> None:
    for tr in tracks:
        tlwh = tr.get("tlwh")
        if not tlwh or len(tlwh) != 4:
            continue
        x, y, w, h = tlwh
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        track_id = int(tr.get("track_id", -1))
        score = float(tr.get("score", 0.0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"id={track_id} s={score:.2f}",
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def draw_roi_boxes(frame, rois: Sequence[ROI]) -> None:
    for (x, y, w, h) in rois:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 1)


def main() -> int:
    cfg = parse_args()

    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index={cfg.camera}")

    if cfg.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
    if cfg.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
    if cfg.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(cfg.fps))

    bytetrack = ByteTracker(frame_rate=cfg.fps, track_buffer=cfg.track_buffer)

    opencv_trackers = []
    paused = False

    last_t = time.time()
    fps_smooth: Optional[float] = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            rois = update_roi_trackers(opencv_trackers, frame)

            # Convert ROI boxes into "detections" for BYTETrack
            dets: List[Det] = [(float(x), float(y), float(w), float(h), 0.99) for (x, y, w, h) in rois]
            tracks = bytetrack.update(dets)

            draw_roi_boxes(frame, rois)
            draw_byte_tracks(frame, tracks)

            now = time.time()
            dt = max(now - last_t, 1e-6)
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
            last_t = now

            cv2.putText(
                frame,
                f"OpenCV ROI trackers: {len(opencv_trackers)} | BYTE tracks: {len(tracks)} | FPS: {fps_smooth:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Keys: s=select ROI  c=clear  p=pause  q/ESC=quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("ROI -> BYTETrack (webcam)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("c"):
            opencv_trackers.clear()
        if key == ord("s"):
            # Grab a fresh frame for selection (avoid selecting on stale)
            ok, frame = cap.read()
            if ok and frame is not None:
                rois = select_rois(frame, multi=cfg.multi_select)
                opencv_trackers = init_roi_trackers(frame, rois, cfg.tracker_name)
        # time.sleep(1/30)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
