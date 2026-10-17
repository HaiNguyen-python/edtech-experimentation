import numpy as np
import pytest
from scipy import stats

from xp.stats import (
    analytic_power,
    cuped,
    holm,
    mde,
    ratio_difference,
    required_n_per_arm,
    srm_chi_square,
    welch_arrays,
    wilson,
)


def test_welch_matches_scipy_including_ci():
    rng = np.random.default_rng(1)
    t, c = rng.normal(1.1, 2.0, 300), rng.normal(1.0, 1.0, 500)
    est = welch_arrays(t, c)
    ref = stats.ttest_ind(t, c, equal_var=False)
    assert est.p_value == pytest.approx(ref.pvalue)
    low, high = ref.confidence_interval(0.95)
    assert (est.ci_low, est.ci_high) == (pytest.approx(low), pytest.approx(high))
    assert est.rel == pytest.approx(t.mean() / c.mean() - 1)
    assert est.rel_ci_low < est.rel < est.rel_ci_high


def test_cuped_keeps_the_effect_and_cuts_variance_by_rho_squared():
    rng = np.random.default_rng(2)
    n = 20_000
    x = rng.normal(0, 1, n)
    treated = rng.random(n) < 0.5
    y = 0.8 * x + rng.normal(0, 0.6, n) + 0.05 * treated  # corr(x, y) = 0.8
    res = cuped(y, x, treated)
    assert res is not None
    assert res.theta == pytest.approx(0.8, abs=0.02)
    assert res.variance_reduction == pytest.approx(0.64, abs=0.02)
    plain = welch_arrays(y[treated], y[~treated])
    assert res.estimate.ci_high - res.estimate.ci_low < 0.65 * (plain.ci_high - plain.ci_low)
    assert res.estimate.ci_low < 0.05 < res.estimate.ci_high


def test_cuped_is_skipped_without_a_pre_period():
    y = np.arange(10.0)
    assert cuped(y, np.zeros(10), np.arange(10) % 2 == 0) is None


def test_ratio_difference_is_a_z_test():
    est = ratio_difference(0.82, 0.0001, 0.80, 0.0001)
    assert est.diff == pytest.approx(0.02)
    assert est.p_value == pytest.approx(2 * stats.norm.sf(0.02 / np.sqrt(0.0002)))


def test_holm_step_down():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


def test_power_formulas_are_consistent():
    var = 8.0
    n = required_n_per_arm(var, effect=0.09)
    assert mde(var, n, n) <= 0.09 < mde(var, n - 50, n - 50)
    assert analytic_power(var, 0.09, n) == pytest.approx(0.8, abs=0.005)


def test_srm_chi_square_and_wilson():
    assert srm_chi_square([5000, 5000], [0.5, 0.5]) == (0.0, 1.0)
    chi2, p = srm_chi_square([6630, 6170], [0.5, 0.5])
    assert chi2 == pytest.approx(2 * 230**2 / 6400) and p < 0.001
    low, high = wilson(50, 1000)
    assert low < 0.05 < high and wilson(0, 100)[0] == pytest.approx(0.0, abs=1e-12)
