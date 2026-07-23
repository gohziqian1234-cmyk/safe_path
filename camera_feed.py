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
