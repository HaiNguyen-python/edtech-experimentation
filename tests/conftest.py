from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from xp import synth
from xp.config import Experiment


@pytest.fixture(scope="session")
def study_data(tmp_path_factory) -> Path:
    """The exact logs behind docs/results.md (default seed, full size: about 2 seconds).

    Written once as Parquet and shared, so the tests check the published decisions.
    """
    out = tmp_path_factory.mktemp("data")
    synth.write(synth.generate(seed=synth.DEFAULT_SEED, scale=1.0), out)
    return out


def make_experiment(**overrides) -> Experiment:
    base = dict(
        name="tiny",
        title="Tiny",
        hypothesis="-",
        layer="home",
        salt="tiny-salt",
        buckets=(0, 1000),
        variants={"control": 0.5, "treatment": 0.5},
        start=datetime(2026, 1, 10),
        ramp_days=2,
        window_days=7,
        primary="lessons_completed_per_week",
        secondary=("completion_rate", "active_days"),
        guardrails=("unsubscribe_rate", "crash_rate"),
        planned_mde=0.05,
        analysis="fixed",
    )
    base.update(overrides)
    return Experiment(**base)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


@pytest.fixture
def tiny_logs() -> dict[str, pd.DataFrame]:
    """Hand-made logs where every per-user metric can be checked by hand.

    a: control, duplicate exposure, pre-period and window events, one event just after the
       window and one before the pre-period (both must be ignored), one crash.
    b: treatment, no events at all (must appear with zeros).
    c: treatment, two lessons on the same day (active_days = 1), unsubscribed.
    d: only exposed in another experiment (excluded).
    e: control, exposed too late for its window to end by as_of (excluded).
    """
    users = pd.DataFrame(
        {"user_id": list("abcde"), "platform": ["ios", "android", "web", "ios", "android"]}
    )
    exposures = pd.DataFrame(
        [
            ("tiny", "a", "control", ts("2026-01-10 08:00")),
            ("tiny", "a", "control", ts("2026-01-11 09:00")),
            ("tiny", "b", "treatment", ts("2026-01-11 00:00")),
            ("tiny", "c", "treatment", ts("2026-01-11 12:00")),
            ("other", "d", "treatment", ts("2026-01-11 12:00")),
            ("tiny", "e", "control", ts("2026-01-20 00:00")),
        ],
        columns=["experiment", "user_id", "variant", "exposed_at"],
    )
    events = pd.DataFrame(
        [
            ("a", ts("2025-12-20 10:00"), "lesson_complete"),  # before the 14-day pre-period
            ("a", ts("2026-01-05 10:00"), "lesson_complete"),  # pre-period
            ("a", ts("2026-01-10 09:00"), "lesson_start"),
            ("a", ts("2026-01-10 09:10"), "lesson_complete"),
            ("a", ts("2026-01-12 10:00"), "lesson_start"),
            ("a", ts("2026-01-12 10:10"), "lesson_complete"),
            ("a", ts("2026-01-13 10:00"), "lesson_start"),  # abandoned
            ("a", ts("2026-01-11 11:00"), "app_crash"),
            ("a", ts("2026-01-17 09:00"), "lesson_complete"),  # after the 7-day window
            ("c", ts("2026-01-12 10:00"), "lesson_start"),
            ("c", ts("2026-01-12 10:10"), "lesson_complete"),
            ("c", ts("2026-01-12 18:00"), "lesson_start"),
            ("c", ts("2026-01-12 18:10"), "lesson_complete"),
            ("c", ts("2026-01-13 08:00"), "notif_unsubscribe"),
            ("d", ts("2026-01-12 10:10"), "lesson_complete"),
            ("e", ts("2026-01-21 10:10"), "lesson_complete"),
        ],
        columns=["user_id", "event_time", "event_name"],
    )
    return {"users": users, "exposures": exposures, "events": events}
