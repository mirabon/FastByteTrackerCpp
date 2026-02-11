from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

DetectionTuple5 = Tuple[float, float, float, float, float]
DetectionTuple6 = Tuple[float, float, float, float, int, float]
DetectionDict = Dict[str, Any]
Detection = Union[DetectionTuple5, DetectionTuple6, DetectionDict]

class Tracker:
    def __init__(self, frame_rate: int = 30, track_buffer: int = 30) -> None: ...
    def update(self, detections: Iterable[Detection]) -> List[Dict[str, Any]]: ...
