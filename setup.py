from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup, find_packages


def _candidate_eigen_dirs() -> list[str]:
    env = os.environ.get("EIGEN3_INCLUDE_DIR")
    candidates = []
    if env:
        candidates.append(env)

    # Common system paths
    candidates += [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",  # Apple Silicon Homebrew
        "/usr/include",  # sometimes Eigen is directly in /usr/include/Eigen
        "/usr/local/include",
    ]

    # Conda prefix, if any
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates += [
            str(Path(conda_prefix) / "include" / "eigen3"),
            str(Path(conda_prefix) / "include"),
        ]

    # Keep only existing directories (setuptools wants paths, not guesses)
    return [p for p in candidates if p and Path(p).is_dir()]


native_inc = "bytetrack_cpp/native/include"
native_src = "bytetrack_cpp/native/src"

sources = [
    "bytetrack_cpp/_core.cpp",
    f"{native_src}/BYTETracker.cpp",
    f"{native_src}/STrack.cpp",
    f"{native_src}/kalmanFilter.cpp",
    f"{native_src}/lapjv.cpp",
    f"{native_src}/utils.cpp",
]

include_dirs = [native_inc] + _candidate_eigen_dirs()

ext_modules = [
    Extension(
        "bytetrack_cpp._core",
        sources=sources,  # MUST be relative paths
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    include_package_data=True,
)
