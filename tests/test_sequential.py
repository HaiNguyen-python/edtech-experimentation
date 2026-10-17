import numpy as np
import pandas as pd
import pytest
from scipy import stats

from xp.sequential import always_valid_p, halfwidth, log_lambda, monitor


@pytest.mark.parametrize("var,tau2", [(0.01, 0.0225), (0.002, 0.01), (1.0, 0.1)])
def test_confidence_sequence_edge_is_exactly_the_rejection_boundary(var, tau2):
    hw = float(halfwidth(var, tau2, alpha=0.05))
    assert float(log_lambda(hw, var, tau2)) == pytest.approx(np.log(1 / 0.05))
    assert float(log_lambda(0.99 * hw, var, tau2)) < np.log(20)


def test_mixture_ratio_known_value_and_shape():
    # d = 0: Lambda = sqrt(V / (V + tau^2)) < 1, so no evidence at all
    assert float(np.exp(log_lambda(0.0, 1.0, 3.0))) == pytest.approx(0.5)
    d = np.linspace(0, 1, 11)
    assert np.all(np.diff(log_lambda(d, 0.01, 0.02)) > 0)


def test_confidence_sequence_is_wider_than_a_fixed_ci():
    var = np.array([0.04, 0.01, 0.0025])
    naive = stats.norm.ppf(0.975) * np.sqrt(var)
    assert np.all(halfwidth(var, 0.01) > naive)


def test_always_valid_p_never_increases():
    diffs = np.array([0.0, 0.3, 0.1, 0.5, 0.0])
    p = always_valid_p(diffs, np.full(5, 0.01), 0.01)
    assert np.all(np.diff(p) <= 0) and p[0] == 1.0


def test_monitor_builds_one_row_per_look():
    looks = pd.DataFrame(
        {
            "day": [1, 1, 2, 2],
            "variant": ["control", "treatment"] * 2,
            "n": [100, 100, 200, 200],
            "mean": [3.0, 3.2, 3.0, 3.1],
            "variance": [4.0, 4.0, 4.0, 4.0],
        }
    )
    out = monitor(looks, "control", "treatment", tau=0.15)
    assert list(out["day"]) == [1, 2]
    assert out["diff"].tolist() == pytest.approx([0.2, 0.1])
    assert np.all(out["cs_low"] < out["naive_low"])
