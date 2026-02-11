# file: webcam_template_bytetrack_test.py
"""
Manual ROI selection + template matching (cv2.matchTemplate) + feed boxes into bytetrack_cpp.

Deps:
  python -m pip install opencv-python

Run:
  python webcam_template_bytetrack_test.py --camera 0

Keys:
  s       select ROI again
  p       pause/resume
  q / ESC quit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from bytetrack_cpp import Tracker as ByteTracker

ROI = Tuple[int, int, int, int]                 # x, y, w, h
Det = Tuple[float, float, float, float, float]  # x, y, w, h, score


@dataclass(frozen=True)
class Config:
    camera: int
    width: int
    height: int
    fps: int
    track_buffer: int
    score_thresh: float
    search_margin: float
    update_template: bool
    template_update_thresh: float


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30, help="Passed into BYTETrack (best-effort)")
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--score-thresh", type=float, default=0.55, help="min matchTemplate score to accept bbox")
    p.add_argument(
        "--search-margin",
        type=float,
        default=2.5,
        help="search window margin in template-sizes around last bbox (0=full-frame search)",
    )
    p.add_argument(
        "--update-template",
        action="store_true",
        help="slowly adapt template when confidence is high (helps with lighting changes)",
    )
    p.add_argument("--template-update-thresh", type=float, default=0.80)
    a = p.parse_args()
    return Config(
        camera=a.camera,
        width=a.width,
        height=a.height,
        fps=a.fps,
        track_buffer=a.track_buffer,
        score_thresh=float(a.score_thresh),
        search_margin=float(a.search_margin),
        update_template=bool(a.update_template),
        template_update_thresh=float(a.template_update_thresh),
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


def select_roi(frame_bgr) -> Optional[ROI]:
    win = "Select ROI (ENTER/SPACE confirm, ESC cancel)"
    cv2.imshow(win, frame_bgr)
    cv2.waitKey(1)
    x, y, w, h = cv2.selectROI(win, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def clamp_roi_to_frame(roi: ROI, w_img: int, h_img: int) -> Optional[ROI]:
    x, y, w, h = roi
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    if w <= 1 or h <= 1:
        return None
    return x, y, w, h


def crop_search_window(
    frame_gray,
    template_w: int,
    template_h: int,
    last_bbox: Optional[ROI],
    margin: float,
) -> Tuple["cv2.Mat", Tuple[int, int]]:
    h_img, w_img = frame_gray.shape[:2]
    if last_bbox is None or margin <= 0:
        return frame_gray, (0, 0)

    lx, ly, lw, lh = last_bbox
    cx = lx + lw / 2.0
    cy = ly + lh / 2.0

    half_w = margin * template_w
    half_h = margin * template_h

    x1 = int(max(0, cx - half_w))
    y1 = int(max(0, cy - half_h))
    x2 = int(min(w_img, cx + half_w))
    y2 = int(min(h_img, cy + half_h))

    if (x2 - x1) < template_w or (y2 - y1) < template_h:
        return frame_gray, (0, 0)

    return frame_gray[y1:y2, x1:x2], (x1, y1)


def match_template(
    frame_gray,
    template_gray,
    last_bbox: Optional[ROI],
    margin: float,
) -> Tuple[Optional[ROI], float]:
    th, tw = template_gray.shape[:2]
    search_gray, (ox, oy) = crop_search_window(frame_gray, tw, th, last_bbox, margin)

    sh, sw = search_gray.shape[:2]
    if sw < tw or sh < th:
        return None, 0.0

    res = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    x = int(max_loc[0] + ox)
    y = int(max_loc[1] + oy)
    bbox = (x, y, int(tw), int(th))
    return bbox, float(max_val)


def blend_template(old_t, new_t, alpha: float = 0.15):
    # why: mild adaptation helps with lighting changes without drifting too fast
    return cv2.addWeighted(old_t, 1.0 - alpha, new_t, alpha, 0.0)


def draw_bbox(frame_bgr, bbox: ROI, score: float, color: Tuple[int, int, int]) -> None:
    x, y, w, h = bbox
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame_bgr,
        f"match={score:.2f}",
        (x, max(0, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def draw_byte_tracks(frame_bgr, tracks: List[dict]) -> None:
    for tr in tracks:
        tlwh = tr.get("tlwh")
        if not tlwh or len(tlwh) != 4:
            continue
        x, y, w, h = tlwh
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        track_id = int(tr.get("track_id", -1))
        score = float(tr.get("score", 0.0))

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"id={track_id} s={score:.2f}",
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def main() -> int:
    cfg = parse_args()
    cap = open_camera(cfg)

    bytetrack = ByteTracker(frame_rate=cfg.fps, track_buffer=cfg.track_buffer)

    template_gray = None
    template_bbox: Optional[ROI] = None
    last_bbox: Optional[ROI] = None

    paused = False
    last_t = time.time()
    fps_smooth: Optional[float] = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h_img, w_img = frame.shape[:2]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            dets: List[Det] = []
            match_bbox: Optional[ROI] = None
            match_score = 0.0

            if template_gray is not None:
                match_bbox, match_score = match_template(
                    frame_gray=frame_gray,
                    template_gray=template_gray,
                    last_bbox=last_bbox,
                    margin=cfg.search_margin,
                )
                if match_bbox is not None:
                    match_bbox = clamp_roi_to_frame(match_bbox, w_img, h_img)
                if match_bbox is not None and match_score >= cfg.score_thresh:
                    last_bbox = match_bbox
                    dets = [(float(match_bbox[0]), float(match_bbox[1]), float(match_bbox[2]), float(match_bbox[3]), float(match_score))]

                    if cfg.update_template and match_score >= cfg.template_update_thresh:
                        x, y, w, h = match_bbox
                        new_patch = frame_gray[y : y + h, x : x + w]
                        if new_patch.shape == template_gray.shape:
                            template_gray = blend_template(template_gray, new_patch)

            tracks = bytetrack.update(dets)

            if template_gray is None:
                cv2.putText(
                    frame,
                    "Press 's' to select ROI",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
            else:
                if match_bbox is not None:
                    draw_bbox(frame, match_bbox, match_score, (255, 200, 0))
                draw_byte_tracks(frame, tracks)

            now = time.time()
            dt = max(now - last_t, 1e-6)
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
            last_t = now

            cv2.putText(
                frame,
                f"BYTE tracks: {len(tracks)} | FPS: {fps_smooth:.1f} | thresh={cfg.score_thresh:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Keys: s=select ROI  p=pause  q/ESC=quit",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow("TemplateMatch -> BYTETrack (webcam)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("s"):
            ok, frame = cap.read()
            if ok and frame is not None:
                roi = select_roi(frame)
                if roi is not None:
                    h_img, w_img = frame.shape[:2]
                    roi = clamp_roi_to_frame(roi, w_img, h_img)
                    if roi is not None:
                        x, y, w, h = roi
                        template_bbox = roi
                        last_bbox = roi
                        template_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y : y + h, x : x + w]

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
