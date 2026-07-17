"""Local CSV and image records for high-risk SafePath events."""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

from risk_engine import RiskAssessment


class EventStore:
    FIELDNAMES = (
        "timestamp",
        "hazard",
        "risk",
        "person_in_zone",
        "hazard_in_zone",
        "warning_issued",
        "snapshot",
    )

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.snapshot_directory = self.output_directory / "snapshots"
        self.event_file = self.output_directory / "events.csv"
        self._lock = threading.Lock()

    def record(
        self,
        assessment: RiskAssessment,
        annotated_frame=None,
        warning_issued: bool = False,
    ) -> dict[str, str]:
        now = datetime.now().astimezone()
        snapshot_name = ""

        with self._lock:
            self.output_directory.mkdir(parents=True, exist_ok=True)

            if annotated_frame is not None:
                import cv2

                self.snapshot_directory.mkdir(parents=True, exist_ok=True)
                snapshot_name = f"event-{now.strftime('%Y%m%d-%H%M%S-%f')}.jpg"
                snapshot_path = self.snapshot_directory / snapshot_name
                if not cv2.imwrite(str(snapshot_path), annotated_frame):
                    snapshot_name = ""

            row = {
                "timestamp": now.isoformat(timespec="seconds"),
                "hazard": assessment.latest_hazard,
                "risk": assessment.risk_level,
                "person_in_zone": "Yes" if assessment.person_in_zone else "No",
                "hazard_in_zone": "Yes" if assessment.hazard_in_zone else "No",
                "warning_issued": "Yes" if warning_issued else "No",
                "snapshot": snapshot_name,
            }
            write_header = not self.event_file.exists()
            with self.event_file.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        return row

    def read_recent(self, limit: int = 10) -> list[dict[str, str]]:
        if not self.event_file.exists():
            return []
        with self._lock:
            with self.event_file.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        return list(reversed(rows[-limit:]))
