import unittest

import numpy as np

from detection import LocalDetector, ModelRuntime


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[104.0, 52.0, 208.0, 156.0]])
        self.cls = FakeTensor([0.0])
        self.conf = FakeTensor([0.90])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    names = {0: "person", 1: "backpack"}

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult()]


class DetectorOptimizationTests(unittest.TestCase):
    def test_runtime_warms_once_and_forces_cpu_imgsz(self):
        model = FakeModel()
        runtime = ModelRuntime(
            "fake.pt",
            inference_size=416,
            model_factory=lambda _path: model,
        )

        self.assertTrue(runtime.warmed_up)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["source"].shape, (416, 416, 3))
        self.assertEqual(model.calls[0]["imgsz"], 416)
        self.assertEqual(model.calls[0]["device"], "cpu")

    def test_detector_downscales_and_restores_original_box_coordinates(self):
        model = FakeModel()
        runtime = ModelRuntime(
            "fake.pt",
            inference_size=416,
            warm_up=False,
            model_factory=lambda _path: model,
        )
        detector = LocalDetector(
            model_path="fake.pt",
            confidence=0.40,
            hazard_labels={"backpack"},
            inference_size=416,
            model_runtime=runtime,
        )
        frame = np.zeros((400, 800, 3), dtype=np.uint8)

        _annotated, assessment = detector.process_frame(frame)

        self.assertEqual(model.calls[0]["source"].shape, (208, 416, 3))
        restored = assessment.detections[0].box
        self.assertEqual(restored, (200.0, 100.0, 400.0, 300.0))
        self.assertEqual(model.calls[0]["imgsz"], 416)

    def test_shared_runtime_does_not_reload_model_per_detector(self):
        model = FakeModel()
        factory_calls = []

        def factory(path):
            factory_calls.append(path)
            return model

        runtime = ModelRuntime(
            "fake.pt",
            inference_size=320,
            warm_up=False,
            model_factory=factory,
        )
        first = LocalDetector(
            "fake.pt",
            model_runtime=runtime,
            hazard_labels={"backpack"},
        )
        second = LocalDetector(
            "fake.pt",
            model_runtime=runtime,
            hazard_labels={"backpack"},
        )

        self.assertIs(first.runtime, second.runtime)
        self.assertEqual(factory_calls, ["fake.pt"])


if __name__ == "__main__":
    unittest.main()
