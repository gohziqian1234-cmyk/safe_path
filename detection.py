"""Local YOLO detection and webcam runner for SafePath AI."""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Iterable

from event_store import EventStore
from risk_engine import DEFAULT_HAZARD_LABELS, Detection, RiskAssessment, RiskEngine
from voice_alert import VoiceAlert


class LocalDetector:
    """Run a pretrained YOLO model and apply SafePath's danger-zone rule."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.40,
        hazard_labels: Iterable[str] = DEFAULT_HAZARD_LABELS,
    ) -> None:
        # Keep Ultralytics settings inside the project instead of writing to
        # the user's roaming AppData directory.
        config_directory = Path(__file__).parent / "outputs" / ".ultralytics"
        config_directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_directory))
        matplotlib_directory = Path(__file__).parent / "outputs" / ".matplotlib"
        matplotlib_directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_directory))

        from ultralytics import YOLO

        self.model_path = model_path
        self.confidence = confidence
        self.risk_engine = RiskEngine(hazard_labels)
        self.model = YOLO(model_path)
        self._prediction_lock = threading.Lock()
        self.target_class_ids = self._find_target_class_ids()

        if not self.target_class_ids:
            raise ValueError(
                "The selected model has no 'person' or configured hazard classes."
            )

    def _find_target_class_ids(self) -> list[int]:
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))

        wanted = {"person", *self.risk_engine.hazard_labels}
        return [
            int(class_id)
            for class_id, label in names.items()
            if str(label).casefold() in wanted
        ]

    def process_frame(self, frame):
        """Return the annotated frame and its structured risk assessment."""

        # Streamlit caches this model across sessions. Serialize inference so
        # simultaneous browser streams do not mutate the same YOLO object.
        with self._prediction_lock:
            results = self.model.predict(
                source=frame,
                classes=self.target_class_ids,
                conf=self.confidence,
                verbose=False,
            )
        detections = self._parse_detections(results[0])
        frame_height, frame_width = frame.shape[:2]
        assessment = self.risk_engine.assess(
            detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        annotated = annotate_frame(frame, assessment)
        return annotated, assessment

    def _parse_detections(self, result) -> list[Detection]:
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        names = self.model.names

        detections: list[Detection] = []
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            label = names[int(class_id)]
            detections.append(
                Detection(
                    label=str(label),
                    confidence=float(confidence),
                    box=tuple(float(value) for value in box),
                )
            )
        return detections


def annotate_frame(frame, assessment: RiskAssessment):
    """Draw the walking zone, model boxes, and current risk on a frame."""

    import cv2
    import numpy as np

    output = frame.copy()
    polygon = np.array(assessment.zone_points, dtype=np.int32)

    overlay = output.copy()
    zone_color = (40, 40, 220) if assessment.risk_level == "HIGH" else (40, 180, 40)
    cv2.fillPoly(overlay, [polygon], zone_color)
    cv2.addWeighted(overlay, 0.16, output, 0.84, 0, output)
    cv2.polylines(output, [polygon], True, zone_color, 2)
    cv2.putText(
        output,
        "WALKING DANGER ZONE",
        (assessment.zone_points[1][0], max(25, assessment.zone_points[1][1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        zone_color,
        2,
        cv2.LINE_AA,
    )

    for detection in assessment.detections:
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        if detection.label.casefold() == "person":
            color = (0, 165, 255)
        elif detection.in_danger_zone:
            color = (30, 30, 235)
        else:
            color = (70, 200, 70)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        zone_marker = " IN ZONE" if detection.in_danger_zone else ""
        label = f"{detection.label} {detection.confidence:.0%}{zone_marker}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    banner_color = (30, 30, 230) if assessment.risk_level == "HIGH" else (35, 150, 35)
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), banner_color, -1)
    cv2.putText(
        output,
        f"SAFEPATH RISK: {assessment.risk_level}",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def open_camera(camera_index: int):
    """Open a webcam, preferring DirectShow on Windows."""

    import cv2

    if os.name == "nt":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if camera.isOpened():
            return camera
        camera.release()

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Camera {camera_index} could not be opened. Close Zoom/Teams or try index 1."
        )
    return camera


def run_camera(
    camera_index: int = 0,
    model_path: str = "yolo11n.pt",
    confidence: float = 0.40,
    voice_enabled: bool = True,
    cooldown_seconds: float = 8.0,
) -> None:
    """Run SafePath in a normal OpenCV desktop window."""

    import cv2

    detector = LocalDetector(model_path, confidence)
    camera = open_camera(camera_index)
    alerts = VoiceAlert(cooldown_seconds=cooldown_seconds, enabled=voice_enabled)
    events = EventStore(Path(__file__).parent / "outputs")

    print("SafePath camera started. Press Q to stop.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read camera frame.")
                break

            annotated, assessment = detector.process_frame(frame)
            if assessment.risk_level == "HIGH":
                warning_issued = alerts.trigger(assessment.warning_message)
                if warning_issued:
                    events.record(assessment, annotated, warning_issued=True)

            cv2.imshow("SafePath AI - Local Safety Monitor", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        alerts.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SafePath AI on a local webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights file")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--cooldown", type=float, default=8.0)
    parser.add_argument("--no-voice", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_camera(
        camera_index=arguments.camera,
        model_path=arguments.model,
        confidence=arguments.confidence,
        voice_enabled=not arguments.no_voice,
        cooldown_seconds=arguments.cooldown,
    )
