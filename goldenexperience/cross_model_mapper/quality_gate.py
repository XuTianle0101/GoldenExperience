"""Quality gate helpers for safe KV reuse."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QualityGate:
    """Threshold gate for mapped KV payloads."""

    min_quality_score: float = 0.95
    max_perplexity_drift: float | None = None

    def accepts(self, quality_score: float, perplexity_drift: float | None = None) -> bool:
        if quality_score < self.min_quality_score:
            return False
        if (
            self.max_perplexity_drift is not None
            and perplexity_drift is not None
            and perplexity_drift > self.max_perplexity_drift
        ):
            return False
        return True

