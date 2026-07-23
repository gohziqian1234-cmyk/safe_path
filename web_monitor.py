"""Thread-safe bridge between browser video frames and SafePath detection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
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
    last_processing_ms: float = 0.0
    average_processing_ms: float = 0.0
    received_frames: int = 0
    dropped_frames: int = 0
    analysis_busy: bool = False
    last_capture_to_result_ms: float = 0.0
    average_capture_to_result_ms: float = 0.0
    last_result_at: float = 0.0
    last_video_callback_ms: float = 0.0
    last_annotation_age_ms: float = 0.0
    last_snapshot_roundtrip_ms: float = 0.0
    assessment_frame_size: tuple[int, int] | None = None


class BrowserMonitor:
    """Analyze only the latest WebRTC frame while video keeps flowing."""

    def __init__(
        self,
        detector: Detector,
        event_store: EventRecorder,
        cooldown_seconds: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.detector = detector
        self.event_store = event_store
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._clock = clock
        self._timer = timer
        self._processing_lock = threading.Lock()
        self._inference_state_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._snapshot = MonitorSnapshot()
        self._last_warning_at = float("-inf")
        self._inference_running = False

    @staticmethod
    def _frame_size(image) -> tuple[int, int] | None:
        shape = getattr(image, "shape", None)
        if not shape or len(shape) < 2:
            return None
        return int(shape[1]), int(shape[0])

    def process_image(self, image, *, captured_at: float | None = None):
        """Process one BGR image and update the shared monitor state."""

        with self._processing_lock:
            processing_started = self._timer()
            if captured_at is None:
                captured_at = processing_started
            annotated, assessment = self.detector.process_frame(image)
            processing_finished = self._timer()
            processing_ms = max(
                0.0,
                (processing_finished - processing_started) * 1000.0,
            )
            capture_to_result_ms = max(
                0.0,
                (processing_finished - captured_at) * 1000.0,
            )
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
                if previous.processed_frames:
                    average_processing_ms = (
                        0.80 * previous.average_processing_ms
                        + 0.20 * processing_ms
                    )
                    average_capture_to_result_ms = (
                        0.80 * previous.average_capture_to_result_ms
                        + 0.20 * capture_to_result_ms
                    )
                else:
                    average_processing_ms = processing_ms
                    average_capture_to_result_ms = capture_to_result_ms
                self._snapshot = replace(
                    previous,
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
                    last_processing_ms=processing_ms,
                    average_processing_ms=average_processing_ms,
                    last_capture_to_result_ms=capture_to_result_ms,
                    average_capture_to_result_ms=average_capture_to_result_ms,
                    last_result_at=processing_finished,
                    assessment_frame_size=self._frame_size(image),
                )
            return annotated

    def _reserve_inference(self) -> bool:
        """Reserve the one allowed inference slot; never create a backlog."""

        with self._inference_state_lock:
            if self._inference_running:
                return False
            self._inference_running = True
            return True

    def _record_received_frame(self, accepted: bool) -> None:
        with self._state_lock:
            previous = self._snapshot
            self._snapshot = replace(
                previous,
                received_frames=previous.received_frames + 1,
                dropped_frames=(
                    previous.dropped_frames + (0 if accepted else 1)
                ),
                analysis_busy=True if accepted else previous.analysis_busy,
            )

    def _record_error(self, error: Exception) -> None:
        with self._state_lock:
            self._snapshot = replace(self._snapshot, last_error=str(error))

    def _run_latest_inference(self, image, captured_at: float) -> None:
        try:
            self.process_image(image, captured_at=captured_at)
        except Exception as error:  # Keep the live track alive.
            self._record_error(error)
        finally:
            with self._inference_state_lock:
                self._inference_running = False
            with self._state_lock:
                self._snapshot = replace(
                    self._snapshot,
                    analysis_busy=False,
                )

    def _start_latest_inference(self, image, captured_at: float) -> bool:
        accepted = self._reserve_inference()
        self._record_received_frame(accepted)
        if not accepted:
            return False

        worker = threading.Thread(
            target=self._run_latest_inference,
            args=(image.copy(), captured_at),
            daemon=True,
            name="safepath-latest-frame-inference",
        )
        try:
            worker.start()
        except Exception as error:
            with self._inference_state_lock:
                self._inference_running = False
            with self._state_lock:
                self._snapshot = replace(
                    self._snapshot,
                    analysis_busy=False,
                    last_error=str(error),
                )
            return False
        return True

    def process_video_frame(self, frame):
        """Pass through current video and update its overlay asynchronously."""

        try:
            import av
            from detection import annotate_frame

            callback_started = self._timer()
            image = frame.to_ndarray(format="bgr24")
            self._start_latest_inference(image, callback_started)

            snapshot = self.snapshot()
            frame_size = self._frame_size(image)
            if (
                snapshot.assessment is not None
                and snapshot.assessment_frame_size == frame_size
            ):
                output = annotate_frame(image, snapshot.assessment)
                annotation_result_at = snapshot.last_result_at
            else:
                output = image
                annotation_result_at = 0.0

            callback_finished = self._timer()
            with self._state_lock:
                previous = self._snapshot
                annotation_age_ms = (
                    max(
                        0.0,
                        (callback_finished - annotation_result_at) * 1000.0,
                    )
                    if annotation_result_at
                    else 0.0
                )
                self._snapshot = replace(
                    previous,
                    last_video_callback_ms=max(
                        0.0,
                        (callback_finished - callback_started) * 1000.0,
                    ),
                    last_annotation_age_ms=annotation_age_ms,
                )

            return av.VideoFrame.from_ndarray(output, format="bgr24")
        except Exception as error:  # WebRTC should continue so the UI can recover.
            self._record_error(error)
            return frame

    def record_snapshot_roundtrip(
        self,
        captured_at_epoch_ms: float | None,
        *,
        response_at_epoch_ms: float | None = None,
    ) -> None:
        """Record browser capture-to-server-response time for snapshot mode."""

        if captured_at_epoch_ms is None:
            return
        if response_at_epoch_ms is None:
            response_at_epoch_ms = time.time() * 1000.0
        roundtrip_ms = max(
            0.0,
            float(response_at_epoch_ms) - float(captured_at_epoch_ms),
        )
        with self._state_lock:
            self._snapshot = replace(
                self._snapshot,
                last_snapshot_roundtrip_ms=roundtrip_ms,
            )

    def snapshot(self) -> MonitorSnapshot:
        """Return the latest immutable state."""

        with self._state_lock:
            return self._snapshot
