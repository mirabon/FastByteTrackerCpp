#pragma once

// Minimal OpenCV compatibility layer.
// The upstream code only relies on cv::Rect_<float> fields: x, y, width, height.
// This header avoids an OpenCV dependency for building the tracker.

namespace cv {

template <typename T>
struct Rect_ {
    T x;
    T y;
    T width;
    T height;

    Rect_() : x(0), y(0), width(0), height(0) {}
    Rect_(T _x, T _y, T _w, T _h) : x(_x), y(_y), width(_w), height(_h) {}
};

}  // namespace cv
