"""Checks that must pass before anyone looks at the effect: SRM, pre-period balance, power.

Why first: a broken experiment produces confident, wrong numbers. If the split of users is not
what the design said (SRM), the arms are no longer comparable, and no amount of statistics on
the metric can fix that. These checks decide whether the result is trustworthy at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import Experiment
from .stats import mde, required_n_per_arm, srm_chi_square, welch_arrays

SRM_ALPHA = 0.001  # strict: SRM is checked on every experiment, so false alarms must be rare
BALANCE_ALPHA = 0.001


@dataclass(frozen=True)
class SrmCheck:
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    p_value: float
    by_platform: pd.DataFrame  # platform, control, treatment, share_treatment, p_value

    @property
    def flagged(self) -> bool:
        return self.p_value < SRM_ALPHA


@dataclass(frozen=True)
class BalanceCheck:
    pre_period_p: float | None  # None when there is no pre-period (new users)
    platform_mix_p: float

    @property
    def flagged(self) -> bool:
        pre_bad = self.pre_period_p is not None and self.pre_period_p < BALANCE_ALPHA
        return pre_bad or self.platform_mix_p < BALANCE_ALPHA


@dataclass(frozen=True)
class PowerCheck:
    baseline: float
    mde_abs: float
    mde_rel: float
    mde_rel_cuped: float | None
    planned_mde: float
    required_per_arm: int

    @property
    def powered(self) -> bool:
        best = self.mde_rel_cuped if self.mde_rel_cuped is not None else self.mde_rel
        return best <= self.planned_mde


def srm_check(counts: pd.DataFrame, exp: Experiment) -> SrmCheck:
    """counts: (platform, variant, users) from assignment_counts.sql.

    The per-platform split is what turns "something is wrong" into "look at the Android client".
    """
    variants = list(exp.variants)
    weights = [exp.variants[v] for v in variants]
    totals = counts.groupby("variant")["users"].sum()
    observed = {v: int(totals.get(v, 0)) for v in variants}
    chi2, p = srm_chi_square(list(observed.values()), weights)
    total = sum(observed.values())
    rows = []
    for platform, g in counts.groupby("platform"):
        per = g.set_index("variant")["users"]
        obs = [int(per.get(v, 0)) for v in variants]
        rows.append(
            {
                "platform": platform,
                exp.control: obs[0],
                exp.treatment: obs[1],
                "share_treatment": obs[1] / sum(obs),
                "p_value": srm_chi_square(obs, weights)[1],
            }
        )
    return SrmCheck(
        observed=observed,
        expected={v: total * w for v, w in zip(variants, weights, strict=True)},
        chi2=chi2,
        p_value=p,
        by_platform=pd.DataFrame(rows),
    )


def balance_check(users: pd.DataFrame, exp: Experiment, covariate: str | None) -> BalanceCheck:
    """Arms should look alike before the change: same pre-period activity, same platform mix."""
    treated = (users["variant"] == exp.treatment).to_numpy()
    pre_p: float | None = None
    if covariate is not None:
        x = users[covariate].to_numpy(dtype=float)
        if np.var(x) > 0:
            pre_p = welch_arrays(x[treated], x[~treated]).p_value
    table = pd.crosstab(users["platform"], users["variant"])
    platform_p = float(stats.chi2_contingency(table.to_numpy())[1]) if table.shape[0] > 1 else 1.0
    return BalanceCheck(pre_period_p=pre_p, platform_mix_p=platform_p)


def power_check(
    users: pd.DataFrame,
    exp: Experiment,
    column: str,
    variance_reduction: float | None,
) -> PowerCheck:
    """Was this experiment big enough to detect the effect it was planned for?

    Uses the pooled variance of the primary metric and the actual arm sizes. With CUPED the
    effective variance is Var * (1 - reduction), so the same users detect smaller effects.
    """
    treated = (users["variant"] == exp.treatment).to_numpy()
    y = users[column].to_numpy(dtype=float)
    n_t, n_c = int(treated.sum()), int((~treated).sum())
    variance = float(np.var(y, ddof=1))
    baseline = float(y[~treated].mean())
    mde_abs = mde(variance, n_t, n_c)
    cuped_rel = None
    if variance_reduction is not None:
        cuped_rel = mde(variance * (1 - variance_reduction), n_t, n_c) / baseline
    return PowerCheck(
        baseline=baseline,
        mde_abs=mde_abs,
        mde_rel=mde_abs / baseline,
        mde_rel_cuped=cuped_rel,
        planned_mde=exp.planned_mde,
        required_per_arm=required_n_per_arm(variance, exp.planned_mde * baseline),
    )
