"""Low-latency browser camera component for SafePath AI."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import streamlit.components.v1 as components


_FRONTEND_DIRECTORY = Path(__file__).parent / "frontend"
_camera_component = components.declare_component(
    "safepath_camera",
    path=str(_FRONTEND_DIRECTORY),
)


@dataclass(frozen=True)
class CameraCapture:
    """One camera-component update."""

    status: str
    image: BytesIO | None = None
    requested_facing_mode: str = "environment"
    active_facing_mode: str = ""
    error: str = ""
    captured_at_epoch_ms: float | None = None


def camera_capture(
    *,
    interval_ms: int = 300,
    width: int = 640,
    height: int = 480,
    jpeg_quality: float = 0.68,
    facing_mode: str = "environment",
    acknowledgement: int = 0,
    key: str | None = None,
) -> CameraCapture | None:
    """Capture back-pressured JPEG frames using the requested phone camera."""

    value = _camera_component(
        intervalMs=max(200, int(interval_ms)),
        width=max(160, int(width)),
        height=max(120, int(height)),
        jpegQuality=min(0.90, max(0.40, float(jpeg_quality))),
        facingMode=(
            "environment" if facing_mode == "environment" else "user"
        ),
        acknowledgement=int(acknowledgement),
        key=key,
        default=None,
    )
    if not isinstance(value, dict):
        return None

    image = None
    data_url = value.get("image")
    if isinstance(data_url, str) and "," in data_url:
        _prefix, encoded = data_url.split(",", 1)
        image = BytesIO(base64.b64decode(encoded))

    raw_capture_time = value.get("capturedAtEpochMs")
    try:
        captured_at_epoch_ms = float(raw_capture_time)
        if not math.isfinite(captured_at_epoch_ms):
            captured_at_epoch_ms = None
    except (TypeError, ValueError):
        captured_at_epoch_ms = None

    return CameraCapture(
        status=str(value.get("status") or "starting"),
        image=image,
        requested_facing_mode=str(
            value.get("requestedFacingMode") or facing_mode
        ),
        active_facing_mode=str(value.get("activeFacingMode") or ""),
        error=str(value.get("error") or ""),
        captured_at_epoch_ms=captured_at_epoch_ms,
    )
