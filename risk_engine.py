"""Pure-Python spatial risk logic for SafePath AI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence


DEFAULT_HAZARD_LABELS = frozenset(
    {
        "backpack",
        "bottle",
        "chair",
        "handbag",
        "sports ball",
        "suitcase",
    }
)


@dataclass(frozen=True)
class Detection:
    """One model detection in pixel coordinates."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]
    in_danger_zone: bool = False

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)


@dataclass(frozen=True)
class DangerZone:
    """A normalized trapezoid representing the likely walking path."""

    normalized_points: tuple[tuple[float, float], ...] = (
        (0.12, 0.98),
        (0.38, 0.48),
        (0.62, 0.48),
        (0.88, 0.98),
    )

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[tuple[int, int], ...]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive.")

        return tuple(
            (
                round(x * (frame_width - 1)),
                round(y * (frame_height - 1)),
            )
            for x, y in self.normalized_points
        )

    def contains(self, point: tuple[float, float], frame_width: int, frame_height: int) -> bool:
        return point_in_polygon(point, self.to_pixels(frame_width, frame_height))


@dataclass(frozen=True)
class RiskAssessment:
    detections: tuple[Detection, ...]
    zone_points: tuple[tuple[int, int], ...]
    person_detected: bool
    person_in_zone: bool
    hazard_detected: bool
    hazard_in_zone: bool
    hazard_names: tuple[str, ...]
    latest_hazard: str
    risk_level: str
    warning_message: str


class RiskEngine:
    """Evaluate detections using the MVP person-plus-hazard rule."""

    def __init__(
        self,
        hazard_labels: Iterable[str] = DEFAULT_HAZARD_LABELS,
        danger_zone: DangerZone | None = None,
    ) -> None:
        self.hazard_labels = {label.casefold() for label in hazard_labels}
        self.danger_zone = danger_zone or DangerZone()

    def assess(
        self,
        detections: Iterable[Detection],
        frame_width: int,
        frame_height: int,
    ) -> RiskAssessment:
        zone_points = self.danger_zone.to_pixels(frame_width, frame_height)
        evaluated = tuple(
            replace(
                detection,
                in_danger_zone=self.danger_zone.contains(
                    detection.bottom_center,
                    frame_width,
                    frame_height,
                ),
            )
            for detection in detections
        )

        people = [item for item in evaluated if item.label.casefold() == "person"]
        hazards = [
            item for item in evaluated if item.label.casefold() in self.hazard_labels
        ]
        people_in_zone = [item for item in people if item.in_danger_zone]
        hazards_in_zone = [item for item in hazards if item.in_danger_zone]

        preferred_hazards = hazards_in_zone or hazards
        latest_hazard = "None"
        if preferred_hazards:
            latest_hazard = max(
                preferred_hazards,
                key=lambda item: item.confidence,
            ).label

        is_high_risk = bool(people_in_zone and hazards_in_zone)
        warning = ""
        if is_high_risk:
            warning = f"Warning. {latest_hazard} detected in the walking path."

        return RiskAssessment(
            detections=evaluated,
            zone_points=zone_points,
            person_detected=bool(people),
            person_in_zone=bool(people_in_zone),
            hazard_detected=bool(hazards),
            hazard_in_zone=bool(hazards_in_zone),
            hazard_names=tuple(sorted({item.label for item in hazards})),
            latest_hazard=latest_hazard,
            risk_level="HIGH" if is_high_risk else "LOW",
            warning_message=warning,
        )


def point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[int, int]],
) -> bool:
    """Return True when a point lies inside or on a polygon boundary."""

    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    previous = polygon[-1]

    for current in polygon:
        x1, y1 = previous
        x2, y2 = current

        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9:
            within_x = min(x1, x2) <= x <= max(x1, x2)
            within_y = min(y1, y2) <= y <= max(y1, y2)
            if within_x and within_y:
                return True

        crosses_ray = (y1 > y) != (y2 > y)
        if crosses_ray:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside

        previous = current

    return inside
