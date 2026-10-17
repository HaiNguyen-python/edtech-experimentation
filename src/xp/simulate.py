"""Monte Carlo checks that the statistics behave as claimed, on the same synthetic user model.

Each function runs many simulated experiments at once as numpy arrays of shape (reps, users),
so thousands of experiments take seconds. Users are drawn exactly like the generator draws
them (`synth.draw_rates`, `synth.draw_lessons`), so these numbers describe this world.

- `fixed_horizon`: A/A false-positive rate for Welch and CUPED, CI widths, bias under a lift.
- `peeking`: daily looks for 28 days; plain t-test vs mSPRT, false-positive rate by look.
- `power_curve`: share of experiments that detect a +3% lift, by users per arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .sequential import log_lambda
from .stats import ALPHA, analytic_power, wilson
from .synth import Behaviour, draw_lessons, draw_rates

MAX_CELLS = 2_000_000  # values per array per chunk: keeps memory small on a shared machine


@dataclass(frozen=True)
class Rate:
    """A simulated rate with its Wilson 95% interval."""

    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.k, self.n)


def _users(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    lift: float,
    weeks: float,
    b: Behaviour,
) -> tuple[np.ndarray, np.ndarray]:
    """(pre-period lessons/week, in-experiment lessons/week) for an array of users."""
    lam = draw_rates(rng, shape, b)
    pre_weeks = b.pre_days / 7
    pre = draw_lessons(rng, lam, pre_weeks, 0.0, b) / pre_weeks
    post = draw_lessons(rng, lam, weeks, lift, b) / weeks
    return pre, post


def _welch_rows(yt: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise Welch test: diff, standard error, two-sided p (one experiment per row)."""
    n_t, n_c = yt.shape[-1], yc.shape[-1]
    vt, vc = yt.var(axis=-1, ddof=1) / n_t, yc.var(axis=-1, ddof=1) / n_c
    se = np.sqrt(vt + vc)
    df = (vt + vc) ** 2 / (vt**2 / (n_t - 1) + vc**2 / (n_c - 1))
    diff = yt.mean(axis=-1) - yc.mean(axis=-1)
    return diff, se, 2 * stats.t.sf(np.abs(diff) / se, df)


def _cuped_rows(
    yt: np.ndarray, yc: np.ndarray, xt: np.ndarray, xc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise CUPED with theta pooled over both arms, as in `stats.cuped`."""
    y = np.concatenate([yt, yc], axis=-1)
    x = np.concatenate([xt, xc], axis=-1)
    xm, ym = x.mean(axis=-1, keepdims=True), y.mean(axis=-1, keepdims=True)
    theta = ((x - xm) * (y - ym)).sum(axis=-1) / ((x - xm) ** 2).sum(axis=-1)
    theta = theta[:, None]
    diff, se, p = _welch_rows(yt - theta * (xt - xm), yc - theta * (xc - xm))
    reduction = 1 - (y - theta * (x - xm)).var(axis=-1, ddof=1) / y.var(axis=-1, ddof=1)
    return diff, se, p, reduction


@dataclass(frozen=True)
class FixedHorizon:
    reps: int
    n_per_arm: int
    lift: float
    baseline: float
    plain_reject: Rate
    cuped_reject: Rate
    plain_mean_diff: float
    cuped_mean_diff: float
    true_diff: float
    plain_ci_width: float
    cuped_ci_width: float
    variance_reduction: float
    plain_sd: float  # spread of the estimated difference across simulated experiments
    cuped_sd: float

    @property
    def width_reduction(self) -> float:
        return 1 - self.cuped_ci_width / self.plain_ci_width

    def mc_se(self, sd: float) -> float:
        """Monte Carlo standard error of a mean estimated difference."""
        return float(sd / np.sqrt(self.reps))


def fixed_horizon(
    reps: int,
    n_per_arm: int,
    lift: float = 0.0,
    seed: int = 0,
    weeks: float = 2.0,
    b: Behaviour | None = None,
) -> FixedHorizon:
    """Many experiments analysed once at the end, with Welch and with CUPED.

    With lift=0 the rejection rate is the false-positive rate (should be about 5%). With a
    lift, the mean estimated difference should equal the true difference for both methods
    (CUPED is unbiased), and CUPED's CIs should be narrower.
    """
    b = b or Behaviour()
    rng = np.random.default_rng(seed)
    chunk = max(1, MAX_CELLS // n_per_arm)
    diffs: dict[str, list[np.ndarray]] = {"plain": [], "cuped": []}
    pvals: dict[str, list[np.ndarray]] = {"plain": [], "cuped": []}
    widths: dict[str, list[np.ndarray]] = {"plain": [], "cuped": []}
    reductions: list[np.ndarray] = []
    z = stats.norm.ppf(1 - ALPHA / 2)
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        xc, yc = _users(rng, (r, n_per_arm), 0.0, weeks, b)
        xt, yt = _users(rng, (r, n_per_arm), lift, weeks, b)
        d, se, p = _welch_rows(yt, yc)
        diffs["plain"].append(d)
        pvals["plain"].append(p)
        widths["plain"].append(2 * z * se)
        d, se, p, red = _cuped_rows(yt, yc, xt, xc)
        diffs["cuped"].append(d)
        pvals["cuped"].append(p)
        widths["cuped"].append(2 * z * se)
        reductions.append(red)
        done += r
    d_all = {k: np.concatenate(v) for k, v in diffs.items()}
    p_all = {k: np.concatenate(v) for k, v in pvals.items()}
    w_all = {k: float(np.concatenate(v).mean()) for k, v in widths.items()}
    baseline = b.lam_shape * b.lam_scale
    return FixedHorizon(
        reps=reps,
        n_per_arm=n_per_arm,
        lift=lift,
        baseline=baseline,
        plain_reject=Rate(int((p_all["plain"] < ALPHA).sum()), reps),
        cuped_reject=Rate(int((p_all["cuped"] < ALPHA).sum()), reps),
        plain_mean_diff=float(d_all["plain"].mean()),
        cuped_mean_diff=float(d_all["cuped"].mean()),
        true_diff=baseline * lift,
        plain_ci_width=w_all["plain"],
        cuped_ci_width=w_all["cuped"],
        variance_reduction=float(np.concatenate(reductions).mean()),
        plain_sd=float(d_all["plain"].std(ddof=1)),
        cuped_sd=float(d_all["cuped"].std(ddof=1)),
    )


@dataclass(frozen=True)
class Peeking:
    reps: int
    days: int
    users_per_day: int  # per arm
    ttest_by_look: list[Rate]  # cumulative: rejected at ANY look up to and including this one
    msprt_by_look: list[Rate]
    ttest_final_only: Rate

    @property
    def ttest_any(self) -> Rate:
        return self.ttest_by_look[-1]

    @property
    def msprt_any(self) -> Rate:
        return self.msprt_by_look[-1]


def peeking(
    reps: int,
    days: int = 28,
    users_per_day: int = 230,
    tau_rel: float = 0.05,
    seed: int = 1,
    b: Behaviour | None = None,
) -> Peeking:
    """A/A experiments looked at every day; users enter each day and are followed one week.

    The default of 230 users per arm per day matches home_redesign (about 13,000 users over 28
    days). At look k we have all users who entered on days 1..k. Cumulative sums give the
    mean and variance at every look without recomputing from scratch.
    """
    b = b or Behaviour()
    rng = np.random.default_rng(seed)
    tau2 = (tau_rel * b.lam_shape * b.lam_scale) ** 2
    z_hits = np.zeros((reps, days), dtype=bool)
    m_hits = np.zeros((reps, days), dtype=bool)
    chunk = max(1, MAX_CELLS // (days * users_per_day))
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        stats_by_arm = []
        for _ in range(2):
            _, y = _users(rng, (r, days, users_per_day), 0.0, 1.0, b)
            s1 = np.cumsum(y.sum(axis=-1), axis=1)
            s2 = np.cumsum((y**2).sum(axis=-1), axis=1)
            n = users_per_day * np.arange(1, days + 1)
            mean = s1 / n
            var = (s2 - n * mean**2) / (n - 1)
            stats_by_arm.append((mean, var / n))
        (mt, vt), (mc, vc) = stats_by_arm
        diff, v = mt - mc, vt + vc
        n_arm = users_per_day * np.arange(1, days + 1)
        df = v**2 / (vt**2 / (n_arm - 1) + vc**2 / (n_arm - 1))
        p = 2 * stats.t.sf(np.abs(diff) / np.sqrt(v), df)
        z_hits[done : done + r] = p < ALPHA
        m_hits[done : done + r] = log_lambda(diff, v, tau2) >= np.log(1 / ALPHA)
        done += r
    z_any = np.logical_or.accumulate(z_hits, axis=1)
    m_any = np.logical_or.accumulate(m_hits, axis=1)
    return Peeking(
        reps=reps,
        days=days,
        users_per_day=users_per_day,
        ttest_by_look=[Rate(int(z_any[:, k].sum()), reps) for k in range(days)],
        msprt_by_look=[Rate(int(m_any[:, k].sum()), reps) for k in range(days)],
        ttest_final_only=Rate(int(z_hits[:, -1].sum()), reps),
    )


@dataclass(frozen=True)
class PowerPoint:
    n_per_arm: int
    plain: Rate
    cuped: Rate
    analytic: float


def power_curve(
    sizes: list[int], reps: int, lift: float = 0.03, seed: int = 2, b: Behaviour | None = None
) -> list[PowerPoint]:
    """Share of simulated experiments with p < 0.05 for a true +3% lift, by users per arm."""
    b = b or Behaviour()
    out = []
    for i, n in enumerate(sizes):
        res = fixed_horizon(reps, n, lift, seed=seed + i, b=b)
        variance = (res.plain_ci_width / (2 * stats.norm.ppf(1 - ALPHA / 2))) ** 2 * n / 2
        out.append(
            PowerPoint(
                n, res.plain_reject, res.cuped_reject, analytic_power(variance, res.true_diff, n)
            )
        )
    return out
