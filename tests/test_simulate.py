from xp.simulate import fixed_horizon, peeking, power_curve


def test_aa_false_positive_rate_is_about_five_percent_for_ttest_and_cuped():
    res = fixed_horizon(reps=800, n_per_arm=400, lift=0.0, seed=3)
    for rate in (res.plain_reject, res.cuped_reject):
        low, high = rate.ci
        assert low < 0.05 < high or 0.025 < rate.value < 0.08
    assert 0.2 < res.variance_reduction < 0.5
    assert res.cuped_ci_width < res.plain_ci_width


def test_cuped_is_unbiased_under_a_real_lift():
    res = fixed_horizon(reps=600, n_per_arm=1000, lift=0.05, seed=4)
    for mean, sd in ((res.plain_mean_diff, res.plain_sd), (res.cuped_mean_diff, res.cuped_sd)):
        assert abs(mean - res.true_diff) < 4 * res.mc_se(sd)
    assert res.cuped_sd < res.plain_sd


def test_daily_peeking_inflates_ttest_but_not_msprt():
    res = peeking(reps=600, days=20, users_per_day=50, seed=5)
    assert res.ttest_any.value > 0.12
    assert res.msprt_any.value <= 0.05
    by_look = [r.value for r in res.ttest_by_look]
    assert by_look == sorted(by_look)  # "rejected at any look so far" can only grow


def test_power_grows_with_sample_size_and_cuped_helps():
    pts = power_curve([500, 4000], reps=200, lift=0.05)
    assert pts[0].plain.value < pts[1].plain.value
    assert pts[1].cuped.value >= pts[1].plain.value - 0.05
    assert abs(pts[1].plain.value - pts[1].analytic) < 0.12
