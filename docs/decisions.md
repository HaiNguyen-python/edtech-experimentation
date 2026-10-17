# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs).

## 1. A seeded generator with planted stories as the ground truth

- **Decision:** synthetic users, exposures and events with five known effects; real logs only
  through the same three-table schema.
- **Why:** there is no public A/B test log with pre-period data, and CI has no internet. A
  generator gives what no real dataset has: the true effect. Every decision in the report can be
  checked against it, and the simulations can use the same user model.
- **Cost:** the world is as simple as I made it (no novelty effect, no weekday pattern, Poisson
  lessons). The seed (60) was chosen from a scan so that home_redesign shows the peeking trap;
  the README says so. Results describe the code, not real learners.

## 2. Hash-based assignment with two salts

- **Decision:** `sha256(salt:user_id)`, first 8 bytes as a number in [0, 1). The layer salt gives
  a bucket (0-999); the experiment salt gives the variant.
- **Why:** deterministic and stateless: the app, the backend and the analysis agree without a
  lookup table, on every device, forever. sha256 is slower than murmur3 but is in the standard
  library and has no bias worth measuring at 1000 buckets.
- **Cost:** changing a salt reshuffles everyone, so salts are part of the experiment's identity
  and must never be edited mid-test. A pinned-value test guards the hash function itself.

## 3. Layers for mutual exclusion

- **Decision:** experiments in the same layer own disjoint bucket ranges; layers are independent.
- **Why:** three home-screen tests could interact (a badge on a redesigned screen), so one user
  sees at most one of them. The reminder lives in another layer, so it can run on 100% of users
  at the same time without stealing traffic.
- **Cost:** experiments in one layer share traffic: home_redesign only gets 40% of the layer.
  Interactions across layers are assumed small and not measured.

## 4. Analyse exposed users from their first exposure, with a fixed window

- **Decision:** the cohort is users with an exposure row; outcomes count from first exposure
  for exactly `window_days`; users whose window has not ended are left out.
- **Why:** users who never saw the change only add noise. A fixed window per user stops early
  users (long follow-up) and late users (short follow-up) from being mixed in one average.
- **Cost:** exposure logging must be correct in the client, which is exactly what failed in
  streak_badge. That is why the SRM check exists. Users entering late are analysed later.

## 5. Metric logic in SQL, driven by a YAML catalog

- **Decision:** per-user metrics, mean summaries, ratio metrics with delta-method variance,
  SRM counts and daily looks are SQL files run by DuckDB; Python binds parameters.
- **Why:** in a company these queries run in the warehouse, and analysts can read and review
  SQL. DuckDB runs the same SQL in-process for tests. The catalog keeps one definition per
  metric instead of one per notebook.
- **Cost:** metric column names must be formatted into SQL (parameters cannot be
  identifiers). They are validated against `^[a-z_][a-z0-9_]*$` at load time and again before
  formatting; a test tries an injection string.

## 6. Checks before results, with "investigate" as a real outcome

- **Decision:** SRM (p < 0.001) or a failed balance check stops the analysis at "investigate";
  estimates are still printed but labelled as not trustworthy.
- **Why:** an SRM means the arms differ in who is in them, so any difference in outcomes may be
  that, not the change. In streak_badge the bug moved Android users from treatment to control.
- **Cost:** p < 0.001 can miss a small SRM in a small experiment. Per-platform SRM helps find
  where it comes from, but more segments mean more chance alarms, so they are shown, not used to
  decide.

## 7. Welch t-test, plus CUPED with a pooled theta

- **Decision:** Welch for means; CUPED on pre-period lessons per week; theta estimated on both
  arms together.
- **Why:** Welch is safe when a treatment changes the spread. CUPED removes the part of the
  noise that was already there before the test (about 34% of the variance here), which is
  worth the same as about 50% more users. A pooled theta keeps the adjustment identical in both
  arms, so the estimator stays unbiased (the simulation checks: 0.0897 vs a true 0.0900).
- **Cost:** CUPED needs a pre-period; new users have none (new_onboarding falls back to Welch).
  Estimating theta from the same data adds a tiny bias of order 1/n, negligible here.

## 8. Delta method for ratio metrics

- **Decision:** completion rate = sum(completed) / sum(started) per arm, variance by the delta
  method on per-user sums.
- **Why:** the unit of randomisation is the user, not the lesson. A binomial variance over
  lessons would ignore that one user's lessons are correlated and give falsely narrow CIs.
  The delta method is closed-form, so it runs in SQL, and it matches the bootstrap in the tests.
- **Cost:** it is a large-sample approximation; for a few hundred users or very skewed
  denominators a bootstrap is safer.

## 9. mSPRT for monitored experiments (not O'Brien-Fleming)

- **Decision:** mixture SPRT with a normal prior on the effect (`tau` = planned MDE), always-valid
  p-values and confidence sequences.
- **Why:** people look at dashboards every day. A group-sequential design with O'Brien-Fleming
  spending is valid only for a look schedule fixed in advance; mSPRT is valid at any time, for
  any number of looks. The simulation shows a plain daily t-test reaching 28.7% false positives
  over 28 looks, while mSPRT stays at 1.1%.
- **Cost:** the boundary is wide early and conservative overall (1.1% instead of 5%), so mSPRT
  needs more users than a single fixed-horizon test to detect the same effect. The variance is
  a plug-in estimate, fine at thousands of users per arm.

## 10. Guardrails: significant AND above threshold

- **Decision:** a guardrail fails only if it is significantly worse and its relative harm is above
  the catalog threshold (10%); one of the two is a warning.
- **Why:** blocking on any significant harm would stop launches over tiny real changes; blocking
  on point estimates would stop them over noise in rare events (crash rates around 3% are very
  noisy). new_onboarding fails clearly: crashes +60.3% (95% CI +39.7% to +80.9%).
- **Cost:** a real but noisy harm can pass as a warning in a small test. Non-inferiority tests
  with planned power for guardrails would be stricter; listed under next steps.

## 11. Holm for secondary metrics, and they never decide

- **Decision:** Holm-adjusted p-values for secondaries; the decision uses the primary and the
  guardrails only.
- **Why:** with several secondary metrics, the chance that one is "significant" by luck is well
  above 5%. Choosing the primary metric before the test is what keeps the decision honest.
- **Cost:** a real secondary effect needs a smaller p-value to be reported as significant.

## 12. Vectorised simulations with Wilson intervals

- **Decision:** numpy arrays of shape `(reps, users)` (chunked), the generator's own user
  model, Wilson 95% intervals on every simulated rate.
- **Why:** thousands of experiments in seconds, so the claims "about 5%" and "inflated" come with
  error bars. Tests use a few hundred reps; the report uses 2,000-4,000.
- **Cost:** about 30 seconds for `xp report`. The simulations check the statistics in this
  world, not in every world.
