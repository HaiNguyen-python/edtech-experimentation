import pytest

from xp import pipeline
from xp.analysis import DONT_SHIP, INVESTIGATE, SHIP, analyze, guardrail_status
from xp.config import load_experiments, load_metrics
from xp.stats import Estimate


def est(rel: float, low: float, high: float) -> Estimate:
    # Absolute CI given the same sign as the relative one (control mean 1.0).
    return Estimate(1.0, 1 + rel, rel, 0.01, low, high, 0.01, rel, low, high)


@pytest.fixture(scope="module")
def results(study_data):
    con = pipeline.connect(study_data)
    metrics = load_metrics()
    return {name: analyze(con, exp, metrics) for name, exp in load_experiments().items()}


def test_srm_is_detected_on_the_buggy_experiment_and_traced_to_android(results):
    streak = results["streak_badge"]
    assert streak.srm.flagged and streak.decision == INVESTIGATE
    worst = streak.srm.by_platform.sort_values("p_value").iloc[0]
    assert worst["platform"] == "android" and worst["share_treatment"] < 0.47
    assert "Sample ratio mismatch" in streak.reasons[0]
    for name in ("practice_reminder", "aa_check", "home_redesign", "new_onboarding"):
        assert not results[name].srm.flagged, name


def test_guardrail_violation_blocks_a_winning_primary(results):
    onboarding = results["new_onboarding"]
    crash = next(r for r in onboarding.rows if r.metric.name == "crash_rate")
    assert crash.status == "fail" and onboarding.decision == DONT_SHIP
    assert onboarding.primary_cuped is None  # new users have no pre-period
    assert onboarding.balance.pre_period_p is None


def test_published_decisions_reproduce(results):
    decisions = {name: r.decision for name, r in results.items()}
    assert decisions == {
        "practice_reminder": SHIP,
        "aa_check": DONT_SHIP,
        "streak_badge": INVESTIGATE,
        "home_redesign": DONT_SHIP,
        "new_onboarding": DONT_SHIP,
    }


def test_reminder_cuped_narrows_the_ci_and_covers_the_planted_lift(results):
    reminder = results["practice_reminder"]
    cuped, plain = reminder.primary_cuped, reminder.primary_plain
    assert cuped is not None and cuped.variance_reduction > 0.2
    assert cuped.estimate.ci_high - cuped.estimate.ci_low < plain.ci_high - plain.ci_low
    assert cuped.estimate.rel_ci_low < 0.03 < cuped.estimate.rel_ci_high
    assert not results["aa_check"].primary.significant


def test_monitored_experiment_has_daily_looks_and_valid_p(results):
    home = results["home_redesign"]
    mon = home.monitoring
    assert mon is not None and mon["n"].is_monotonic_increasing
    assert mon["always_valid_p"].is_monotonic_decreasing
    assert home.naive_win_day() is not None  # the planted peeking trap
    assert mon["always_valid_p"].iloc[-1] > 0.05 and home.primary_plain.p_value > 0.05


@pytest.mark.parametrize(
    "rel,low,high,expected",
    [
        (0.60, 0.40, 0.80, "fail"),  # significantly worse and above the 10% threshold
        (0.05, 0.01, 0.09, "warn"),  # significantly worse but small
        (0.20, -0.10, 0.50, "warn"),  # large but not significant
        (-0.20, -0.40, -0.01, "pass"),  # significantly better
    ],
)
def test_guardrail_status_rules(rel, low, high, expected):
    crash = load_metrics()["crash_rate"]  # direction: decrease, threshold 10%
    assert guardrail_status(est(rel, low, high), crash) == expected
