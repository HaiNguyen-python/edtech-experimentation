"""Seeded synthetic logs for five experiments, each with a story planted on purpose.

The sandbox this repo was built in has no real app data, so this module plays the app. It writes
three tables with the schema a real product would log (see `pipeline.load_logs`):

- users(user_id, platform, signup_at)
- exposures(experiment, user_id, variant, exposed_at)   one row per visit, so with duplicates
- events(user_id, event_time, event_name)               lesson_start, lesson_complete,
                                                        notif_unsubscribe, app_crash

How a user behaves: each learner has a personal lesson rate `lam` (lessons per week, Gamma
distributed, mean 3). In any period the rate drifts a little (log-normal noise), and lessons are
Poisson draws from it. Because the same `lam` drives the two weeks before exposure and the weeks
after, pre-period activity predicts in-experiment activity. That is exactly what CUPED uses.

Planted truth (`PLANTED`):
1. practice_reminder: +3% relative lift in lessons, guardrails flat.
2. aa_check: nothing; both arms get the same app.
3. streak_badge: a client bug on Android shows control to 15% of users hashed into treatment
   and logs them as control. Sample ratio mismatch, and the effect is diluted.
4. new_onboarding: +5% lessons, but the crash rate is 1.5x higher.
5. home_redesign: no effect at all, monitored daily for 28 days of entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .assignment import assign_many
from .config import Experiment, load_experiments, load_layers

PLATFORMS = ("ios", "android", "web")
PLATFORM_SHARE = (0.40, 0.45, 0.15)
DEFAULT_SEED = 60
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Behaviour:
    """Knobs of the synthetic world, shared by the generator and the simulations."""

    lam_shape: float = 2.0
    lam_scale: float = 1.5  # mean 3 lessons per week, sd 2.1
    drift_sd: float = 0.35  # period-to-period change in a user's rate
    abandon_ratio: float = 0.25  # abandoned lesson starts per completed lesson
    crash_prob: float = 0.03  # chance of at least one crash in a 14-day window
    unsubscribe_prob: float = 0.02  # chance of turning off notifications in a window
    active_share: float = 0.8  # share of eligible users who open the app during a test
    pre_days: int = 14


@dataclass(frozen=True)
class Plant:
    lift: float = 0.0
    crash_mult: float = 1.0
    unsubscribe_mult: float = 1.0
    android_shown_control: float = 0.0  # the streak_badge client bug


PLANTED: dict[str, Plant] = {
    "practice_reminder": Plant(lift=0.03),
    "aa_check": Plant(),
    "streak_badge": Plant(lift=0.02, android_shown_control=0.15),
    "home_redesign": Plant(),
    "new_onboarding": Plant(lift=0.05, crash_mult=1.5),
}


def draw_rates(rng: np.random.Generator, size: int | tuple[int, ...], b: Behaviour) -> np.ndarray:
    """Each user's long-run lesson rate (lessons per week)."""
    return rng.gamma(b.lam_shape, b.lam_scale, size)


def draw_lessons(
    rng: np.random.Generator,
    lam: np.ndarray,
    weeks: float,
    lift: float | np.ndarray,
    b: Behaviour,
) -> np.ndarray:
    """Completed lessons in one period: the rate drifts, then a Poisson draw.

    The drift is mean-one log-normal noise, so it does not change the average rate. It makes
    pre-period activity an imperfect predictor, as in real apps (otherwise CUPED would look
    unrealistically strong).
    """
    drift = np.exp(rng.normal(-0.5 * b.drift_sd**2, b.drift_sd, lam.shape))
    return rng.poisson(lam * drift * weeks * (1.0 + lift))


def _timestamps(start: np.ndarray, length_days: np.ndarray | float, u: np.ndarray) -> np.ndarray:
    seconds = (u * np.asarray(length_days) * SECONDS_PER_DAY).astype("int64")
    return start + seconds.astype("timedelta64[s]")


@dataclass
class _EventBuffer:
    codes: list[np.ndarray]
    times: list[np.ndarray]
    names: list[np.ndarray]

    def add(self, codes: np.ndarray, times: np.ndarray, name: str) -> None:
        self.codes.append(codes)
        self.times.append(times)
        self.names.append(np.full(len(codes), name, dtype=object))


def _add_period_events(
    rng: np.random.Generator,
    buf: _EventBuffer,
    codes: np.ndarray,
    starts: np.ndarray,
    days: int,
    completes: np.ndarray,
    abandons: np.ndarray,
) -> None:
    """Lesson events for one period: every completion has a start 10 minutes earlier."""
    idx = np.repeat(np.arange(len(codes)), completes)
    t = _timestamps(starts[idx], days, rng.random(len(idx)))
    buf.add(codes[idx], t, "lesson_start")
    buf.add(codes[idx], t + np.timedelta64(600, "s"), "lesson_complete")
    idx = np.repeat(np.arange(len(codes)), abandons)
    buf.add(codes[idx], _timestamps(starts[idx], days, rng.random(len(idx))), "lesson_start")


def _simulate_experiment(
    rng: np.random.Generator,
    exp: Experiment,
    plant: Plant,
    users: pd.DataFrame,
    lam: np.ndarray,
    variants: np.ndarray,
    buf: _EventBuffer,
    b: Behaviour,
) -> pd.DataFrame:
    """Exposures and events of one experiment. Returns its exposure log."""
    in_exp = np.array([v is not None for v in variants])
    if exp.new_users:
        active = in_exp  # a new user is exposed at sign-up
    else:
        active = in_exp & (rng.random(len(users)) < b.active_share)
    pos = np.flatnonzero(active)
    codes = pos.astype("int64")
    platform = users["platform"].to_numpy()[pos]
    assigned = variants[pos].astype(str)
    logged = assigned.copy()
    bug = (
        (assigned == exp.treatment)
        & (platform == "android")
        & (rng.random(len(pos)) < plant.android_shown_control)
    )
    logged[bug] = exp.control  # the client shows control and logs control
    treated = logged == exp.treatment

    if exp.new_users:
        first = users["signup_at"].to_numpy()[pos].astype("datetime64[s]")
    else:
        first = _timestamps(
            np.full(len(pos), np.datetime64(exp.start, "s")), exp.ramp_days, rng.random(len(pos))
        )
    # Repeat visits create duplicate exposure rows; the SQL keeps the first one per user.
    visits = 1 + rng.poisson(1.5, len(pos))
    vidx = np.repeat(np.arange(len(pos)), visits)
    later = rng.random(len(vidx)) * exp.window_days
    later[np.r_[0, np.cumsum(visits)[:-1]]] = 0.0  # the first visit is the exposure itself
    exposures = pd.DataFrame(
        {
            "experiment": exp.name,
            "user_id": users["user_id"].to_numpy()[pos][vidx],
            "variant": logged[vidx],
            "exposed_at": _timestamps(first[vidx], later, np.ones(len(vidx))),
        }
    )

    weeks = exp.window_days / 7
    lam_u = lam[pos]
    lift = np.where(treated, plant.lift, 0.0)
    if not exp.new_users:
        pre_start = first - np.timedelta64(b.pre_days * SECONDS_PER_DAY, "s")
        pre_weeks = b.pre_days / 7
        done = draw_lessons(rng, lam_u, pre_weeks, 0.0, b)
        dropped = rng.poisson(lam_u * pre_weeks * b.abandon_ratio)
        _add_period_events(rng, buf, codes, pre_start, b.pre_days, done, dropped)
    done = draw_lessons(rng, lam_u, weeks, lift, b)
    dropped = rng.poisson(lam_u * weeks * (1.0 + lift) * b.abandon_ratio)
    _add_period_events(rng, buf, codes, first, exp.window_days, done, dropped)

    scale = exp.window_days / 14
    p_crash = b.crash_prob * scale * np.where(treated, plant.crash_mult, 1.0)
    p_unsub = b.unsubscribe_prob * scale * np.where(treated, plant.unsubscribe_mult, 1.0)
    for name, prob in (("app_crash", p_crash), ("notif_unsubscribe", p_unsub)):
        hit = rng.random(len(pos)) < prob
        buf.add(codes[hit], _timestamps(first[hit], exp.window_days, rng.random(hit.sum())), name)
    return exposures


def make_users(rng: np.random.Generator, n_existing: int, n_new: int) -> pd.DataFrame:
    """Existing users signed up last year; new users sign up during the onboarding test."""
    onboarding = load_experiments()["new_onboarding"]
    old = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n_existing), "D")
    new_start = np.datetime64(onboarding.start, "s")
    new = _timestamps(np.full(n_new, new_start), onboarding.ramp_days, rng.random(n_new))
    return pd.DataFrame(
        {
            "user_id": [f"u{i:06d}" for i in range(n_existing)]
            + [f"n{i:06d}" for i in range(n_new)],
            "platform": rng.choice(PLATFORMS, n_existing + n_new, p=PLATFORM_SHARE),
            "signup_at": np.concatenate([old.to_numpy().astype("datetime64[s]"), new]),
            "is_new": np.r_[np.zeros(n_existing, bool), np.ones(n_new, bool)],
        }
    )


def generate(
    seed: int = DEFAULT_SEED, scale: float = 1.0, behaviour: Behaviour | None = None
) -> dict[str, pd.DataFrame]:
    """Users, exposures and events for all experiments in the catalog.

    `scale=1` gives 40,000 existing users and 25,000 new sign-ups (about 3 million events).
    Experiments in different calendar periods share the same existing users and the same
    personal lesson rates, like a real app; their analysis windows do not overlap in time.
    """
    b = behaviour or Behaviour()
    rng = np.random.default_rng(seed)
    users = make_users(rng, int(40_000 * scale), int(25_000 * scale))
    lam = draw_rates(rng, len(users), b)
    layers = load_layers()
    buf = _EventBuffer([], [], [])
    exposure_logs = []
    for exp in load_experiments().values():
        eligible = users["is_new"].to_numpy() == exp.new_users
        variants = np.full(len(users), None, dtype=object)
        ids = users["user_id"].to_numpy()[eligible]
        variants[eligible] = assign_many(exp, layers[exp.layer], ids)
        plant = PLANTED.get(exp.name, Plant())
        exposure_logs.append(_simulate_experiment(rng, exp, plant, users, lam, variants, buf, b))
    events = pd.DataFrame(
        {
            "user_id": pd.Categorical.from_codes(
                np.concatenate(buf.codes).astype(np.int64), categories=pd.Index(users["user_id"])
            ),
            "event_time": np.concatenate(buf.times),
            "event_name": pd.Categorical(np.concatenate(buf.names)),
        }
    ).sort_values("event_time", ignore_index=True)
    exposures = pd.concat(exposure_logs, ignore_index=True)
    return {"users": users.drop(columns="is_new"), "exposures": exposures, "events": events}


def write(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
