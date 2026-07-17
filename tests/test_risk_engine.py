import unittest

from risk_engine import Detection, RiskEngine, point_in_polygon


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine(hazard_labels={"backpack", "bottle"})

    def test_high_risk_requires_person_and_hazard_in_zone(self) -> None:
        detections = [
            Detection("person", 0.93, (400, 400, 600, 900)),
            Detection("backpack", 0.88, (450, 700, 550, 880)),
        ]

        result = self.engine.assess(detections, 1000, 1000)

        self.assertEqual(result.risk_level, "HIGH")
        self.assertTrue(result.person_in_zone)
        self.assertTrue(result.hazard_in_zone)
        self.assertEqual(result.latest_hazard, "backpack")
        self.assertIn("backpack", result.warning_message)

    def test_hazard_outside_zone_stays_low_risk(self) -> None:
        detections = [
            Detection("person", 0.93, (400, 400, 600, 900)),
            Detection("bottle", 0.88, (0, 700, 80, 900)),
        ]

        result = self.engine.assess(detections, 1000, 1000)

        self.assertEqual(result.risk_level, "LOW")
        self.assertTrue(result.person_in_zone)
        self.assertFalse(result.hazard_in_zone)

    def test_hazard_without_person_stays_low_risk(self) -> None:
        result = self.engine.assess(
            [Detection("backpack", 0.88, (450, 700, 550, 880))],
            1000,
            1000,
        )

        self.assertEqual(result.risk_level, "LOW")
        self.assertFalse(result.person_detected)
        self.assertTrue(result.hazard_in_zone)

    def test_point_on_polygon_boundary_counts_as_inside(self) -> None:
        square = ((0, 0), (10, 0), (10, 10), (0, 10))
        self.assertTrue(point_in_polygon((0, 5), square))
        self.assertTrue(point_in_polygon((5, 5), square))
        self.assertFalse(point_in_polygon((15, 5), square))


if __name__ == "__main__":
    unittest.main()
