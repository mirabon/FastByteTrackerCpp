#pragma once
// Minimal OpenCV compatibility layer for BYTETrack (types only).
// This avoids requiring OpenCV at build time.

#include <cmath>

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

using Rect2f = Rect_<float>;

struct Scalar {
    double val[4];
    Scalar() : val{0, 0, 0, 0} {}
    Scalar(double v0, double v1 = 0, double v2 = 0, double v3 = 0) : val{v0, v1, v2, v3} {}
};

}  // namespace cv
