# How it works

This page follows one experiment from assignment to decision, then explains the simulations.
File references point to `src/xp/`.

## 1. Catalogs: experiments and metrics (`experiments.yml`, `metrics.yml`, `config.py`)

An experiment is configured, not coded. `experiments.yml` gives each experiment a layer, a
bucket range, variants with weights, a start date, how long users keep entering (`ramp_days`),
how long each user is followed (`window_days`), a primary metric, secondary metrics,
guardrails, the planned minimum detectable effect, and `analysis: fixed` or `sequential`.

`metrics.yml` defines every metric once:

| Field | Meaning |
| --- | --- |
| `type: mean` + `column` | per-user value averaged over users (Welch t-test) |
| `covariate` | pre-period column for CUPED |
| `type: ratio` + `numerator`/`denominator` | `sum(num) / sum(den)` over users (delta method) |
| `direction` | `increase` or `decrease` is good |
| `guardrail_threshold` | largest relative harm accepted (0.10 = 10% worse) |

`config.py` loads both files into frozen dataclasses and refuses bad input: weights that do not
sum to 1, bucket ranges that overlap inside a layer, unknown metrics, and column names that are
not plain identifiers (they are later formatted into SQL, so this is the injection guard).

## 2. Assignment (`assignment.py`)

```
u      = first 8 bytes of sha256(f"{salt}:{user_id}") / 2^64      # uniform in [0, 1)
bucket = floor(u_layer * 1000)                                    # 0..999 per layer
```

1. The **layer hash** puts each user in one of 1000 buckets of the layer. An experiment owns
   a bucket range (streak_badge owns 200-599, so 40% of the layer's traffic). Two experiments
   in the same layer have disjoint ranges, so they never share a user: mutual exclusion.
2. The **experiment hash** (a different salt) picks the variant by walking the cumulative
   weights. Because the salt differs from the layer salt, the variant split is independent of
   which buckets the experiment owns.
3. Different layers have different salts, so being in the reminder test says nothing about
   being in the home-screen tests: experiments in different layers overlap independently.

`ExposureLogger` writes one JSON line per exposure. An exposure is logged when the user
reaches the changed screen, not at assignment: users who never see the change cannot be
affected by it, and including them only dilutes the effect.

## 3. The synthetic app (`synth.py`)

Each learner has a personal rate `lam ~ Gamma(2, 1.5)` lessons per week (mean 3). In any
period the rate drifts by a mean-one log-normal factor (sd 0.35 on the log scale) and
completed lessons are `Poisson(lam * drift * weeks * (1 + lift))`. Abandoned lesson starts are
a further Poisson with 25% of the undrifted rate. Crashes and unsubscribes are rare Bernoulli events.

The same `lam` drives the 14 days before exposure and the analysis window, which is why
pre-period lessons predict in-experiment lessons (correlation about 0.58): the signal CUPED
uses. The drift keeps that correlation realistic.

The five planted stories (`PLANTED`):

| Experiment | Layer | Planted truth |
| --- | --- | --- |
| practice_reminder | notifications (100%) | `lift = 0.03` for treated users |
| aa_check | home 0-199 | nothing |
| streak_badge | home 200-599 | Android client shows control to 15% of treatment users and logs them as control |
| home_redesign | home 600-999 | nothing; users enter over 28 days, 7-day window, monitored daily |
| new_onboarding | onboarding (100%) | new sign-ups only, `lift = 0.05`, crash probability x1.5 |

Users are assigned with the real `assignment.assign`. Each exposed user also gets 1 +
Poisson(1.5) repeat exposure rows, so the log has duplicates like a real one. Experiments run in
calendar periods whose analysis windows do not overlap, and the existing users and their rates
are shared across experiments, as in a real app.

## 4. Per-user metrics in SQL (`sql/user_metrics.sql`)

One query, parameterised by experiment, window, pre-period length and `as_of`:

1. `first_exposure`: per user, the earliest exposure and the variant logged then
   (`arg_min(variant, exposed_at)`). Repeat visits disappear here.
2. `cohort`: only users whose whole window has ended by `as_of`. A user who entered yesterday
   has one day of data; counting them with zeros for the missing days would bias the metric.
3. `user_events`: events from `[exposure - 14 days, exposure + window)`, flagged as pre-period
   or in-window. Nothing before exposure counts as an outcome.
4. `per_user`: `count(*) FILTER (...)` for lessons completed and started, distinct active
   days, any unsubscribe, any crash, and pre-period lessons.
5. The final `LEFT JOIN` keeps exposed users with no events at all, with zeros. Dropping them
   would remove exactly the users a reminder is meant to bring back.

`lessons_completed_per_week = completed * 7 / window_days`, so experiments with different
windows report on the same scale.

## 5. Summaries in SQL (`mean_metrics.sql`, `ratio_metric.sql`, `assignment_counts.sql`)

- `mean_metrics.sql` unpivots every numeric column and returns `n`, `mean` and `var_samp` per
  variant. Those three numbers are all a Welch test needs.
- `ratio_metric.sql` computes the completion rate `R = mean(X) / mean(Y)` per variant with the
  delta-method variance

  `Var(R) = (1/n) [ var(X)/mu_y^2 - 2 mu_x cov(X,Y)/mu_y^3 + mu_x^2 var(Y)/mu_y^4 ]`.

  Lessons of one user are correlated (an engaged user starts and finishes many), and the user
  is the unit of randomisation. Treating every lesson as an independent trial would give a CI
  that is far too narrow. The test suite checks this variance against a 4,000-draw bootstrap.
- `assignment_counts.sql` counts distinct users per platform and variant for the SRM check.

## 6. Checks before results (`checks.py`)

- **SRM**: chi-square goodness of fit of users per variant against the designed weights. Flag at
  p < 0.001 (strict, because it runs on every experiment and a false alarm stops the analysis).
  The same test per platform shows *where* the mismatch is: for streak_badge, Android has 42.4%
  of its users in treatment, iOS and web are fine.
- **Pre-period balance**: Welch test on pre-period lessons and a chi-square test on the platform
  mix between arms. A difference before the change means the groups were not comparable.
- **Power**: `MDE = (z_0.975 + z_0.8) * sqrt(var * (1/n_t + 1/n_c))` with the pooled variance and
  actual arm sizes, also with CUPED's reduced variance, compared with the planned MDE. It also
  reports how many users per arm the planned effect needs.

## 7. Estimates (`stats.py`)

- **Welch t-test** from summary statistics, with Welch-Satterthwaite degrees of freedom. The
  relative lift `mean_t / mean_c - 1` gets a delta-method standard error that includes the noise
  in the control mean.
- **CUPED**: `theta = cov(Y, X) / var(X)` pooled over both arms, `Y_adj = Y - theta (X - mean X)`,
  then Welch on `Y_adj`. Variance falls by about `corr(X, Y)^2`. Because X is measured before
  exposure, its expectation is the same in both arms, so the expected difference does not
  change. For new users there is no pre-period: CUPED returns `None` and the raw test is used.
- **Delta method** comparison of two ratios: a z-test on `R_t - R_c` with variance
  `Var(R_t) + Var(R_c)`.
- **Holm** step-down adjustment across the secondary metrics.

## 8. Monitoring (`sequential.py`, `sql/daily_looks.sql`)

`daily_looks.sql` rebuilds what a dashboard would have shown on each day d: the users whose
window had ended by `start + d`, per variant. `sequential.monitor` turns that into, per day:

- the naive t-test p-value and CI (only valid at one planned look);
- the mixture SPRT statistic
  `Lambda = sqrt(V / (V + tau^2)) * exp(d^2 tau^2 / (2 V (V + tau^2)))`,
  the always-valid p-value `min over earlier looks of min(1, 1/Lambda)`, and the confidence
  sequence `d +/- sqrt(V (V + tau^2) / tau^2 * (2 log(1/alpha) + log((V + tau^2) / V)))`.

Under no effect, Lambda is a non-negative martingale with mean 1, so by Ville's inequality the
chance that it *ever* exceeds `1/alpha = 20` is at most 5%. That is why the analyst may look
every day. `tau` is the planned MDE in metric units (5% of the control mean for home_redesign).

## 9. Decision (`analysis.py`)

In this order:

1. SRM flagged or balance failed: **investigate**. The estimates are shown "for reference only".
2. Sequential experiments: always-valid p at the last look >= 0.05: **don't ship** (and the
   report names the day a naive t-test would have declared a win).
3. A/A tests: a significant primary or Holm-adjusted secondary means investigate; otherwise
   "don't ship" with the note that the platform looks healthy.
4. Any guardrail **fail** (significantly worse AND worse than its threshold): **don't ship**.
   One of the two only is a **warn**, listed but not blocking.
5. Primary (CUPED when available) significantly better: **ship**; worse or flat: **don't ship**.

## 10. Reports (`report.py`)

`write_experiment` writes `docs/experiments/<name>.md`: decision and reasons, design, checks
table, SRM by platform, metric table (raw primary, CUPED primary, secondaries with Holm p,
guardrails with status), CUPED theta and variance reduction, daily monitoring for sequential
tests, and a chart (a forest plot of relative changes, or the naive CI vs the confidence
sequence over days). `xp report` does this for all five and adds `docs/results.md` with the
simulations and two charts.

## 11. Simulations (`simulate.py`)

All simulations draw users with the generator's own functions and run many experiments at once
as numpy arrays of shape `(reps, users)` (chunked to stay under 2 million values per array).

- `fixed_horizon`: Welch and CUPED computed row-wise. With `lift = 0` the rejection rate is the
  false-positive rate. With `lift = 0.03` the mean estimate across experiments is compared with
  the true difference (`3 lessons * 0.03 = 0.09`), with a Monte Carlo standard error.
- `peeking`: `(reps, days, users_per_day)` per arm; cumulative sums over days give the mean and
  variance at every look in one pass. A test counts as a false positive if it rejects at any
  look so far, for the t-test and for mSPRT.
- `power_curve`: `fixed_horizon` at several sizes, next to the analytic normal-approximation
  power.

Every rate is reported with a Wilson 95% interval, so "about 5%" is a claim with an error bar.
