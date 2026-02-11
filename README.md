# Fast bytetrack-cpp

A small, pip-installable Python package that builds a C++17 extension wrapping a minimal BYTETrack implementation.

## Install

You need Eigen headers (`libeigen3-dev` on Ubuntu/Debian).

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install .
```

## Usage

```python
from bytetrack_cpp import Tracker

t = Tracker(frame_rate=30, track_buffer=30)
tracks = t.update([(100, 50, 80, 120, 0.9)])
```
