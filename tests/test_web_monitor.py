import unittest
from threading import Event
from time import monotonic, sleep

import av
import numpy as np

from risk_engine import Detection, RiskEngine
from web_monitor import BrowserMonitor


class FakeDetector:
    def __init__(self, assessment):
        self.assessment = assessment

    def process_frame(self, frame):
        return frame, self.assessment


class FakeEventStore:
    def __init__(self):
        self.records = []

    def record(self, assessment, annotated_frame=None, warning_issued=False):
        self.records.append((assessment, annotated_frame, warning_issued))
        return {}


class BlockingDetector(FakeDetector):
    def __init__(self, assessment, started, release):
        super().__init__(assessment)
        self.started = started
        self.release = release

    def process_frame(self, frame):
        self.started.set()
        self.release.wait(timeout=2.0)
        return super().process_frame(frame)


class BrowserMonitorTests(unittest.TestCase):
    def make_assessment(self, high_risk: bool):
        detections = [Detection("person", 0.95, (400, 400, 600, 900))]
        if high_risk:
            detections.append(Detection("backpack", 0.90, (450, 700, 550, 880)))
        return RiskEngine({"backpack"}).assess(detections, 1000, 1000)

    def test_high_risk_event_obeys_warning_cooldown(self):
        times = iter([10.0, 12.0, 16.0])
        events = FakeEventStore()
        monitor = BrowserMonitor(
            detector=FakeDetector(self.make_assessment(high_risk=True)),
            event_store=events,
            cooldown_seconds=5.0,
            clock=lambda: next(times),
        )

        monitor.process_image("frame-one")
        monitor.process_image("frame-two")
        monitor.process_image("frame-three")

        snapshot = monitor.snapshot()
        self.assertEqual(snapshot.processed_frames, 3)
        self.assertEqual(snapshot.warning_sequence, 2)
        self.assertEqual(len(events.records), 2)
        self.assertIn("backpack", snapshot.warning_message)

    def test_low_risk_frame_does_not_create_event(self):
        events = FakeEventStore()
        monitor = BrowserMonitor(
            detector=FakeDetector(self.make_assessment(high_risk=False)),
            event_store=events,
        )

        monitor.process_image("frame")

        snapshot = monitor.snapshot()
        self.assertEqual(snapshot.assessment.risk_level, "LOW")
        self.assertEqual(snapshot.warning_sequence, 0)
        self.assertEqual(events.records, [])

    def test_processing_latency_uses_exponential_moving_average(self):
        timer_values = iter([1.0, 1.1, 2.0, 2.3])
        monitor = BrowserMonitor(
            detector=FakeDetector(self.make_assessment(high_risk=False)),
            event_store=FakeEventStore(),
            timer=lambda: next(timer_values),
        )

        monitor.process_image("frame-one")
        monitor.process_image("frame-two")

        snapshot = monitor.snapshot()
        self.assertAlmostEqual(snapshot.last_processing_ms, 300.0)
        self.assertAlmostEqual(snapshot.average_processing_ms, 140.0)

    def test_capture_to_result_and_snapshot_roundtrip_are_measured(self):
        timer_values = iter([1.0, 1.25])
        monitor = BrowserMonitor(
            detector=FakeDetector(self.make_assessment(high_risk=False)),
            event_store=FakeEventStore(),
            timer=lambda: next(timer_values),
        )

        monitor.process_image("frame", captured_at=0.80)
        monitor.record_snapshot_roundtrip(
            1000.0,
            response_at_epoch_ms=1234.0,
        )

        snapshot = monitor.snapshot()
        self.assertAlmostEqual(snapshot.last_processing_ms, 250.0)
        self.assertAlmostEqual(snapshot.last_capture_to_result_ms, 450.0)
        self.assertAlmostEqual(snapshot.last_snapshot_roundtrip_ms, 234.0)

    def test_webrtc_drops_frames_while_inference_is_busy(self):
        started = Event()
        release = Event()
        monitor = BrowserMonitor(
            detector=BlockingDetector(
                self.make_assessment(high_risk=False),
                started,
                release,
            ),
            event_store=FakeEventStore(),
        )
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")

        first_started_at = monotonic()
        first_output = monitor.process_video_frame(frame)
        first_elapsed = monotonic() - first_started_at
        self.assertTrue(started.wait(timeout=1.0))

        second_output = monitor.process_video_frame(frame)
        busy_snapshot = monitor.snapshot()
        self.assertLess(first_elapsed, 0.5)
        self.assertEqual(first_output.width, 160)
        self.assertEqual(second_output.height, 120)
        self.assertEqual(busy_snapshot.received_frames, 2)
        self.assertEqual(busy_snapshot.dropped_frames, 1)
        self.assertTrue(busy_snapshot.analysis_busy)

        release.set()
        deadline = monotonic() + 2.0
        while monitor.snapshot().analysis_busy and monotonic() < deadline:
            sleep(0.01)

        finished_snapshot = monitor.snapshot()
        self.assertFalse(finished_snapshot.analysis_busy)
        self.assertEqual(finished_snapshot.processed_frames, 1)


if __name__ == "__main__":
    unittest.main()
