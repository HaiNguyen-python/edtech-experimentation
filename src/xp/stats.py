"""Fixed-horizon statistics: Welch t-test, CUPED, delta method, power/MDE, Holm, SRM, Wilson.

Everything here works on small summaries (n, mean, variance) or on numpy arrays, and the same
functions are used by the experiment analysis and by the simulations. That way the
simulations test the code that produces the reported numbers, not a copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ALPHA = 0.05


@dataclass(frozen=True)
class Estimate:
    """Treatment minus control, absolute and relative, with a two-sided CI and p-value."""

    control: float
    treatment: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    rel: float
    rel_ci_low: float
    rel_ci_high: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def _relative(mean_t: float, se_t: float, mean_c: float, se_c: float) -> tuple[float, float]:
    """Relative lift mean_t/mean_c - 1 and its delta-method standard error.

    The control mean is itself noisy, so dividing the absolute CI by the control mean would
    understate the uncertainty; the second term adds the control's share.
    """
    rel = mean_t / mean_c - 1.0
    se = float(np.sqrt(se_t**2 / mean_c**2 + mean_t**2 * se_c**2 / mean_c**4))
    return rel, se


def welch(
    mean_t: float,
    var_t: float,
    n_t: int,
    mean_c: float,
    var_c: float,
    n_c: int,
    alpha: float = ALPHA,
) -> Estimate:
    """Welch's t-test from summary statistics (unequal variances, Welch-Satterthwaite df).

    Welch instead of Student: a treatment can change the spread as well as the mean (here the
    reminder adds lessons mostly for already-active users), and Welch costs nothing when the
    variances happen to be equal.
    """
    vt, vc = var_t / n_t, var_c / n_c
    se = float(np.sqrt(vt + vc))
    df = (vt + vc) ** 2 / (vt**2 / (n_t - 1) + vc**2 / (n_c - 1))
    diff = mean_t - mean_c
    p = float(2 * stats.t.sf(abs(diff) / se, df)) if se > 0 else 1.0
    q = float(stats.t.ppf(1 - alpha / 2, df))
    rel, rel_se = _relative(mean_t, float(np.sqrt(vt)), mean_c, float(np.sqrt(vc)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    return Estimate(
        control=mean_c,
        treatment=mean_t,
        diff=diff,
        se=se,
        ci_low=diff - q * se,
        ci_high=diff + q * se,
        p_value=p,
        rel=rel,
        rel_ci_low=rel - z * rel_se,
        rel_ci_high=rel + z * rel_se,
    )


def welch_arrays(y_t: np.ndarray, y_c: np.ndarray, alpha: float = ALPHA) -> Estimate:
    return welch(
        float(y_t.mean()),
        float(y_t.var(ddof=1)),
        len(y_t),
        float(y_c.mean()),
        float(y_c.var(ddof=1)),
        len(y_c),
        alpha,
    )


@dataclass(frozen=True)
class CupedResult:
    estimate: Estimate
    theta: float
    variance_reduction: float  # 1 - Var(adjusted) / Var(raw), pooled over both arms


def cuped(
    y: np.ndarray, x: np.ndarray, treated: np.ndarray, alpha: float = ALPHA
) -> CupedResult | None:
    """CUPED: compare Y - theta (X - mean X) instead of Y, with X measured before exposure.

    theta = Cov(Y, X) / Var(X) minimises the variance of the adjusted metric; the variance
    shrinks by the factor (1 - corr(X, Y)^2). X comes from before the experiment, so the
    treatment cannot have changed it, and E[X] is the same in both arms: the adjustment
    changes the noise, not the expected difference. theta is pooled over both arms, which
    keeps it identical for treatment and control.

    Returns None when X has no variance (for example new users with no pre-period).
    """
    var_x = float(np.var(x, ddof=1))
    if var_x <= 0:
        return None
    theta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    adj = y - theta * (x - x.mean())
    est = welch_arrays(adj[treated], adj[~treated], alpha)
    # The relative lift uses the adjusted control mean as base; it equals the raw control mean
    # up to the (random) pre-period difference between arms.
    reduction = 1.0 - float(np.var(adj, ddof=1) / np.var(y, ddof=1))
    return CupedResult(est, theta, reduction)


def ratio_difference(
    ratio_t: float, var_t: float, ratio_c: float, var_c: float, alpha: float = ALPHA
) -> Estimate:
    """Compare two ratio metrics given their delta-method variances (from ratio_metric.sql).

    A z-test: the delta method is a large-sample approximation, so a t correction would add
    precision that is not really there.
    """
    se = float(np.sqrt(var_t + var_c))
    diff = ratio_t - ratio_c
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else 1.0
    rel, rel_se = _relative(ratio_t, float(np.sqrt(var_t)), ratio_c, float(np.sqrt(var_c)))
    return Estimate(
        control=ratio_c,
        treatment=ratio_t,
        diff=diff,
        se=se,
        ci_low=diff - z * se,
        ci_high=diff + z * se,
        p_value=p,
        rel=rel,
        rel_ci_low=rel - z * rel_se,
        rel_ci_high=rel + z * rel_se,
    )


def mde(variance: float, n_t: int, n_c: int, alpha: float = ALPHA, power: float = 0.8) -> float:
    """Minimum detectable absolute effect for a two-sided test at this alpha and power."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(variance * (1 / n_t + 1 / n_c)))


def required_n_per_arm(
    variance: float, effect: float, alpha: float = ALPHA, power: float = 0.8
) -> int:
    """Users per arm (equal split) needed to detect an absolute `effect`: inverse of `mde`."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * variance * (z / effect) ** 2))


def analytic_power(variance: float, effect: float, n_per_arm: int, alpha: float = ALPHA) -> float:
    se = np.sqrt(2 * variance / n_per_arm)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - effect / se) + stats.norm.cdf(-z - effect / se))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls the family-wise error rate).

    Used for secondary metrics: with two or three of them, the chance that at least one is
    "significant" by luck is well above 5%. Holm is uniformly more powerful than Bonferroni and
    needs no independence assumption.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def srm_chi_square(observed: list[int], weights: list[float]) -> tuple[float, float]:
    """Chi-square goodness of fit of the users per variant against the designed split."""
    obs = np.asarray(observed, dtype=float)
    expected = obs.sum() * np.asarray(weights) / np.sum(weights)
    chi2 = float(((obs - expected) ** 2 / expected).sum())
    return chi2, float(stats.chi2.sf(chi2, len(obs) - 1))


def wilson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Wilson score interval for a proportion (used for simulated false-positive rates)."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)
