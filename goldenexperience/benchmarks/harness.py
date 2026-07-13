"""Reusable benchmark harness for synthetic and model-backed experiments."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.metrics import BenchmarkRecord, BenchmarkSummary, summarize


class BenchmarkHarness:
    """Collect timing records and export artifact-friendly summaries."""

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.records: list[BenchmarkRecord] = []

    def time_operation(
        self,
        name: str,
        operation: Callable[[], Any],
        bytes_size: int = 0,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> Any:
        start = time.perf_counter()
        result = operation()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.records.append(
            BenchmarkRecord(
                name=name,
                latency_ms=latency_ms,
                bytes_size=bytes_size,
                cache_hit=result is not None,
                metadata=metadata or {},
            )
        )
        return result

    def summary(self) -> dict[str, BenchmarkSummary]:
        return {
            name: summarize(name, [record for record in self.records if record.name == name])
            for name in sorted({record.name for record in self.records})
        }

    def export_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        output = {
            "experiment": self.experiment_name,
            "summaries": {name: asdict(summary) for name, summary in self.summary().items()},
            "records": [asdict(record) for record in self.records],
            "extra": extra or {},
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    def reset(self) -> None:
        self.records.clear()
