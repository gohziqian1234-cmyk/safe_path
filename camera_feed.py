"""Helpers for decoding frames from the cloud-compatible camera component."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol

import cv2
import numpy as np


class CameraImage(Protocol):
    def getvalue(self) -> bytes: ...


def decode_camera_frame(camera_image: CameraImage) -> tuple[str, np.ndarray]:
    """Return a stable frame digest and a decoded BGR image."""

    encoded = camera_image.getvalue()
    if not encoded:
        raise ValueError("The browser returned an empty camera frame.")

    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The browser returned a camera frame that could not be decoded.")

    return sha256(encoded).hexdigest(), frame


def resize_for_inference(
    frame: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, float, float]:
    """Downscale a frame and return x/y factors for restoring box coordinates."""

    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("Camera frame dimensions must be positive.")

    target_side = max(160, int(max_side))
    longest_side = max(frame_height, frame_width)
    if longest_side <= target_side:
        return frame, 1.0, 1.0

    scale = target_side / float(longest_side)
    resized_width = max(1, round(frame_width * scale))
    resized_height = max(1, round(frame_height * scale))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    return (
        resized,
        frame_width / float(resized_width),
        frame_height / float(resized_height),
    )
