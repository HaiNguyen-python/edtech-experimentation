"""Typed views of the two YAML catalogs: metrics.yml and experiments.yml.

Metric and experiment definitions live in YAML so that product analysts can add a metric or an
experiment without touching Python. This module validates them once at load time, so a typo
fails loudly here instead of producing a silently wrong SQL query later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
MetricType = Literal["mean", "ratio"]
Direction = Literal["increase", "decrease"]


@dataclass(frozen=True)
class Metric:
    name: str
    description: str
    type: MetricType
    direction: Direction
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    covariate: str | None = None
    guardrail_threshold: float | None = None

    def columns(self) -> list[str]:
        """User-metric columns this metric reads (used to validate SQL identifiers)."""
        cols = [self.column, self.numerator, self.denominator, self.covariate]
        return [c for c in cols if c is not None]


@dataclass(frozen=True)
class Layer:
    name: str
    salt: str


@dataclass(frozen=True)
class Experiment:
    name: str
    title: str
    hypothesis: str
    layer: str
    salt: str
    buckets: tuple[int, int]
    variants: dict[str, float]
    start: datetime
    ramp_days: int
    window_days: int
    primary: str
    secondary: tuple[str, ...]
    guardrails: tuple[str, ...]
    planned_mde: float
    analysis: Literal["fixed", "sequential"]
    aa: bool = False
    new_users: bool = False

    @property
    def end(self) -> datetime:
        """Last moment of data: the last user enters at start + ramp and is followed a window."""
        return self.start + timedelta(days=self.ramp_days + self.window_days)

    @property
    def control(self) -> str:
        return next(iter(self.variants))

    @property
    def treatment(self) -> str:
        return list(self.variants)[1]


def _read_yaml(path: Path | None, default_name: str) -> dict[str, Any]:
    if path is None:
        text = resources.files("xp").joinpath(default_name).read_text()
    else:
        text = path.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{default_name}: expected a mapping at the top level")
    return data


def load_metrics(path: Path | None = None) -> dict[str, Metric]:
    """Load and validate the metric catalog (defaults to the packaged metrics.yml)."""
    out: dict[str, Metric] = {}
    for raw in _read_yaml(path, "metrics.yml")["metrics"]:
        metric = Metric(**raw)
        if metric.type == "mean" and metric.column is None:
            raise ValueError(f"metric {metric.name}: a mean metric needs `column`")
        if metric.type == "ratio" and (metric.numerator is None or metric.denominator is None):
            raise ValueError(f"metric {metric.name}: a ratio metric needs numerator/denominator")
        for col in metric.columns():
            if not IDENTIFIER.match(col):
                raise ValueError(f"metric {metric.name}: bad column name {col!r}")
        if metric.name in out:
            raise ValueError(f"duplicate metric {metric.name}")
        out[metric.name] = metric
    return out


def load_layers(path: Path | None = None) -> dict[str, Layer]:
    return {raw["name"]: Layer(**raw) for raw in _read_yaml(path, "experiments.yml")["layers"]}


def load_experiments(
    path: Path | None = None, metrics: dict[str, Metric] | None = None
) -> dict[str, Experiment]:
    """Load experiments and check them against the layers and the metric catalog.

    Checks: weights sum to 1, exactly two variants (the analysis compares two arms), bucket
    ranges inside 0..1000 and not overlapping within a layer (mutual exclusion), and every
    referenced metric exists.
    """
    data = _read_yaml(path, "experiments.yml")
    layers = {raw["name"] for raw in data["layers"]}
    metrics = metrics if metrics is not None else load_metrics()
    out: dict[str, Experiment] = {}
    for raw in data["experiments"]:
        start = raw["start"]
        start_dt = datetime.fromisoformat(start) if isinstance(start, str) else start
        if isinstance(start_dt, date) and not isinstance(start_dt, datetime):
            start_dt = datetime(start_dt.year, start_dt.month, start_dt.day)
        exp = Experiment(
            **{
                **raw,
                "buckets": tuple(raw["buckets"]),
                "secondary": tuple(raw["secondary"]),
                "guardrails": tuple(raw["guardrails"]),
                "start": start_dt,
            }
        )
        if exp.layer not in layers:
            raise ValueError(f"experiment {exp.name}: unknown layer {exp.layer}")
        if len(exp.variants) != 2 or abs(sum(exp.variants.values()) - 1.0) > 1e-9:
            raise ValueError(f"experiment {exp.name}: need two variants with weights summing to 1")
        lo, hi = exp.buckets
        if not 0 <= lo < hi <= 1000:
            raise ValueError(f"experiment {exp.name}: buckets must satisfy 0 <= lo < hi <= 1000")
        for name in (exp.primary, *exp.secondary, *exp.guardrails):
            if name not in metrics:
                raise ValueError(f"experiment {exp.name}: unknown metric {name}")
        for other in out.values():
            if other.layer == exp.layer and lo < other.buckets[1] and other.buckets[0] < hi:
                raise ValueError(f"experiments {other.name} and {exp.name} overlap in a layer")
        out[exp.name] = exp
    return out
