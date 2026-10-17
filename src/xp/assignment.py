"""Deterministic assignment: sha256 hashing, layers for mutual exclusion, exposure logging.

Why hashing instead of a random draw stored in a table: the same user must see the same variant
on every device and every request, without a database lookup. sha256(salt + user_id) gives a
number that is stable forever, independent across experiments (different salts) and uniform.

Two hashes per experiment:
1. layer hash  -> bucket 0..999. An experiment owns a bucket range inside its layer, so two
   experiments in the same layer never share a user (mutual exclusion), and traffic allocation
   is "how many buckets you own".
2. experiment hash -> a number in [0, 1) that picks the variant by its weight. Using a second,
   experiment-specific salt means the variant split is independent of the bucket range.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import Experiment, Layer, load_experiments, load_layers

N_BUCKETS = 1000
_TWO_64 = float(2**64)


def hash_unit(salt: str, unit_id: str) -> float:
    """Map (salt, unit) to a uniform number in [0, 1) using the first 8 bytes of sha256."""
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _TWO_64


def bucket(salt: str, unit_id: str, n_buckets: int = N_BUCKETS) -> int:
    """Bucket 0..n_buckets-1. Bias from the modulo is about n / 2^64: negligible."""
    return int(hash_unit(salt, unit_id) * n_buckets)


def pick_variant(variants: dict[str, float], u: float) -> str:
    """Walk the cumulative weights. Order of `variants` is part of the design: keep it fixed."""
    total = 0.0
    names = list(variants)
    for name in names:
        total += variants[name]
        if u < total:
            return name
    return names[-1]  # u is in [0, 1) and weights sum to 1; this only absorbs rounding


def assign(experiment: Experiment, layer: Layer, user_id: str) -> str | None:
    """Variant for this user, or None when the user's layer bucket is outside the experiment."""
    lo, hi = experiment.buckets
    if not lo <= bucket(layer.salt, user_id) < hi:
        return None
    return pick_variant(experiment.variants, hash_unit(experiment.salt, user_id))


def assign_many(experiment: Experiment, layer: Layer, user_ids: Iterable[str]) -> np.ndarray:
    """Vectorised-friendly helper for the generator: an object array of variants or None."""
    return np.array([assign(experiment, layer, u) for u in user_ids], dtype=object)


@dataclass
class ExposureLogger:
    """Append-only JSON-lines exposure log.

    An exposure is logged when the user actually reaches the changed screen, not when they are
    assigned. Analysing only exposed users keeps the unaffected majority out of the metric,
    which would otherwise dilute the effect (and the power) without changing the truth.
    """

    path: Path | None = None
    records: list[dict[str, str]] = field(default_factory=list)

    def log(self, experiment: str, user_id: str, variant: str, at: datetime | None = None) -> None:
        when = (at or datetime.now(UTC)).isoformat()
        record = {"experiment": experiment, "user_id": user_id, "variant": variant, "at": when}
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")


@dataclass
class Assigner:
    """All experiments and layers from the catalog; assigns a user to every active experiment."""

    experiments: dict[str, Experiment]
    layers: dict[str, Layer]

    @classmethod
    def from_catalog(cls) -> Assigner:
        return cls(load_experiments(), load_layers())

    def assign_all(self, user_id: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for exp in self.experiments.values():
            variant = assign(exp, self.layers[exp.layer], user_id)
            if variant is not None:
                out[exp.name] = variant
        return out
