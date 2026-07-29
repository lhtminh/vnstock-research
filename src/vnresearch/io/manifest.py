"""The snapshot manifest: what the mirror contains and when it was taken.

Every downstream artefact records the manifest it was built from. Without that
you cannot tell a change in results from a change in the data underneath.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    row_counts: dict[str, int]
    adjustment_epochs: list[int]
    max_ingested_at: str | None = None
    as_of: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def id(self) -> str:
        """Short handle for stamping reports and model files."""
        return self.as_of[:19].replace(":", "").replace("-", "").replace("T", "-")

    def check_single_epoch(self) -> None:
        """Refuse to proceed on a mixed-basis price series.

        VCI restates its whole history on a corporate action. Bars from two
        different restatements joined together show a price gap that never
        happened, and every return built across the seam is fiction.
        """
        if len(self.adjustment_epochs) > 1:
            raise ValueError(
                f"daily_prices spans adjustment epochs {self.adjustment_epochs}; "
                "run vn-audit.exe in vnstock-service and repair before modelling"
            )


def write_manifest(mirror_dir: Path, m: Manifest) -> Path:
    path = mirror_dir / MANIFEST_NAME
    path.write_text(json.dumps(asdict(m), indent=2), encoding="utf-8")
    return path


def load_manifest(mirror_dir: Path) -> Manifest:
    path = mirror_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path} — run `vnr mirror` first")
    return Manifest(**json.loads(path.read_text(encoding="utf-8")))
