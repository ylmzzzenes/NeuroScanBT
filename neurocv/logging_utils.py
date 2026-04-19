"""Basit CSV eğitim günlüğü (TensorBoard zorunluluğu olmadan)."""
from __future__ import annotations

import csv
from pathlib import Path


class CsvLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._w: csv.DictWriter | None = None

    def log(self, row: dict) -> None:
        if self._w is None:
            self._w = csv.DictWriter(self._f, fieldnames=list(row.keys()))
            self._w.writeheader()
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()
