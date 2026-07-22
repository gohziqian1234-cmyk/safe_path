import unittest

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


if __name__ == "__main__":
    unittest.main()
