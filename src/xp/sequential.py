"""Always-valid inference for experiments that are watched every day: mixture SPRT (mSPRT).

The problem: a t-test p-value is valid at ONE pre-planned moment. If you look every day and stop
the first time p < 0.05, the chance of a false win under no effect grows with the number of looks
(the simulation in `simulate.py` measures it). Teams look anyway.

Choice: mSPRT with a normal mixture (Johari, Pekelis, Walsh 2017; used by Optimizely), instead of
group-sequential O'Brien-Fleming spending. Why: mSPRT allows looking at ANY time, as often as you
like, without fixing the number or timing of looks in advance. Group-sequential designs need the
look schedule up front and break if people look in between. The cost: if you only ever look once
at the planned end, mSPRT is less powerful than a fixed test (its boundary is wider).

The maths, for the difference in means `d` with variance `V = s_t^2/n_t + s_c^2/n_c` and a
normal prior on the true effect, theta ~ N(0, tau^2):

    Lambda = sqrt(V / (V + tau^2)) * exp( d^2 tau^2 / (2 V (V + tau^2)) )

Lambda is a likelihood ratio averaged over effects. Under no effect it is a non-negative
martingale, so by Ville's inequality P(Lambda ever >= 1/alpha) <= alpha. That is what makes it
safe to monitor continuously. Inverting the boundary gives a confidence sequence, a CI that is
valid at all times at once:

    d +/- sqrt( V (V + tau^2) / tau^2 * (2 log(1/alpha) + log((V + tau^2) / V)) )

tau is a tuning choice: the size of effect we expect. We use the planned MDE in metric units.
The variance is a plug-in estimate, fine at the sample sizes here (thousands per arm).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stats import ALPHA


def log_lambda(diff: np.ndarray | float, var: np.ndarray | float, tau2: float) -> np.ndarray:
    """log of the mixture likelihood ratio (log space avoids overflow for large effects)."""
    d, v = np.asarray(diff, dtype=float), np.asarray(var, dtype=float)
    return 0.5 * np.log(v / (v + tau2)) + d**2 * tau2 / (2 * v * (v + tau2))


def halfwidth(var: np.ndarray | float, tau2: float, alpha: float = ALPHA) -> np.ndarray:
    """Half-width of the confidence sequence: the |diff| at which Lambda equals 1/alpha."""
    v = np.asarray(var, dtype=float)
    return np.sqrt(v * (v + tau2) / tau2 * (2 * np.log(1 / alpha) + np.log((v + tau2) / v)))


def always_valid_p(diffs: np.ndarray, variances: np.ndarray, tau2: float) -> np.ndarray:
    """p_n = min over looks k <= n of min(1, 1/Lambda_k): never goes back up once it drops."""
    p = np.minimum(1.0, np.exp(-log_lambda(diffs, variances, tau2)))
    return np.minimum.accumulate(p, axis=-1)


def monitor(looks: pd.DataFrame, control: str, treatment: str, tau: float) -> pd.DataFrame:
    """One row per look day: the naive t-test a dashboard would show, and the mSPRT view.

    `looks` comes from daily_looks.sql: (day, variant, n, mean, variance).
    """
    wide = looks.pivot(index="day", columns="variant", values=["n", "mean", "variance"])
    n_c, n_t = wide["n"][control].to_numpy(), wide["n"][treatment].to_numpy()
    m_c, m_t = wide["mean"][control].to_numpy(), wide["mean"][treatment].to_numpy()
    var = wide["variance"][treatment].to_numpy() / n_t + wide["variance"][control].to_numpy() / n_c
    diff = m_t - m_c
    se = np.sqrt(var)
    z = stats.norm.ppf(1 - ALPHA / 2)
    hw = halfwidth(var, tau**2)
    return pd.DataFrame(
        {
            "day": wide.index.to_numpy(),
            "n": n_c + n_t,
            "diff": diff,
            "naive_p": 2 * stats.norm.sf(np.abs(diff) / se),
            "naive_low": diff - z * se,
            "naive_high": diff + z * se,
            "cs_low": diff - hw,
            "cs_high": diff + hw,
            "always_valid_p": always_valid_p(diff, var, tau**2),
        }
    )
