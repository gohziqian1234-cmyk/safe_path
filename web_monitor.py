"""Thread-safe bridge between browser video frames and SafePath detection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from risk_engine import RiskAssessment


class Detector(Protocol):
    def process_frame(self, frame): ...


class EventRecorder(Protocol):
    def record(
        self,
        assessment: RiskAssessment,
        annotated_frame=None,
        warning_issued: bool = False,
    ) -> dict[str, str]: ...


@dataclass(frozen=True)
class MonitorSnapshot:
    """Immutable state that the Streamlit UI can safely read."""

    assessment: RiskAssessment | None = None
    warning_sequence: int = 0
    warning_message: str = ""
    processed_frames: int = 0
    last_error: str = ""


class BrowserMonitor:
    """Run detection in the WebRTC worker thread and publish UI state."""

    def __init__(
        self,
        detector: Detector,
        event_store: EventRecorder,
        cooldown_seconds: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.detector = detector
        self.event_store = event_store
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._clock = clock
        self._processing_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._snapshot = MonitorSnapshot()
        self._last_warning_at = float("-inf")

    def process_image(self, image):
        """Process one BGR image and update the shared monitor state."""

        with self._processing_lock:
            annotated, assessment = self.detector.process_frame(image)
            now = self._clock()
            warning_issued = (
                assessment.risk_level == "HIGH"
                and now - self._last_warning_at >= self.cooldown_seconds
            )

            error_message = ""
            if warning_issued:
                self._last_warning_at = now
                try:
                    self.event_store.record(
                        assessment,
                        annotated_frame=annotated,
                        warning_issued=True,
                    )
                except Exception as error:  # Keep the video stream alive.
                    error_message = f"Could not save the event: {error}"

            with self._state_lock:
                previous = self._snapshot
                self._snapshot = MonitorSnapshot(
                    assessment=assessment,
                    warning_sequence=(
                        previous.warning_sequence + 1
                        if warning_issued
                        else previous.warning_sequence
                    ),
                    warning_message=(
                        assessment.warning_message
                        if warning_issued
                        else previous.warning_message
                    ),
                    processed_frames=previous.processed_frames + 1,
                    last_error=error_message,
                )
            return annotated

    def process_video_frame(self, frame):
        """Convert an ``av.VideoFrame``, run AI, and return an annotated frame."""

        try:
            import av

            image = frame.to_ndarray(format="bgr24")
            annotated = self.process_image(image)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as error:  # WebRTC should continue so the UI can recover.
            with self._state_lock:
                previous = self._snapshot
                self._snapshot = MonitorSnapshot(
                    assessment=previous.assessment,
                    warning_sequence=previous.warning_sequence,
                    warning_message=previous.warning_message,
                    processed_frames=previous.processed_frames,
                    last_error=str(error),
                )
            return frame

    def snapshot(self) -> MonitorSnapshot:
        """Return the latest immutable state."""

        with self._state_lock:
            return self._snapshot
