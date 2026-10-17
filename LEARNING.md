# Learning guide (for me, Hai)

This file is my study guide for this repo. Goal: explain every part in an interview, change it
with confidence, and answer "why?" three levels deep. Work through it before pinning the repo.

Start with [docs/how-it-works.md](docs/how-it-works.md) (the code walkthrough and the maths) and
[docs/decisions.md](docs/decisions.md) (the reasons). This file is the practice plan.

## 0. Why this matters to me

At HaiEduTech and BambooTech we change things all the time: a reminder message, a new lesson
screen, a placement quiz. The honest question after each change is "did it help learners, or did
we just get a lucky week?". In a classroom I could see my students. In an app I only see numbers,
and numbers lie in three common ways:

- **Bad randomisation.** If one platform's app is buggy, the groups are not the same people.
  This is like comparing my morning class with my evening class and calling it a teaching method
  effect.
- **Noise.** Some learners do ten lessons a week, some do none. A small real effect hides in that
  spread unless the test is big, or unless I remove the part of the spread I already knew about
  before the test (CUPED: "how active was this learner last month?").
- **Impatience.** Checking the dashboard every morning and stopping when it is green is like
  marking a test after the first three questions. The simulation shows it: 28.7% false wins.

Things I must be careful about when I tell this story: the effects were planted by my
generator, so the repo proves the code finds them and the statistics keep their promises in this
world. It does not prove that a reminder lifts lessons by 3% in a real app.

## 1. Run it and watch each step (1 hour)

```bash
make install
make check                        # lint, types, tests with coverage
xp generate                       # data/*.parquet: open them in DuckDB or pandas
python -c "import duckdb; duckdb.sql(\"SELECT event_name, count(*) FROM 'data/events.parquet' GROUP BY 1\").show()"
xp analyze practice_reminder      # read reports/practice_reminder.md
xp analyze streak_badge           # why "investigate"? find the Android row
xp report                         # docs/results.md and all charts (~30 s)
xp serve                          # http://127.0.0.1:8000/docs, try /assign
```

Then change one thing and predict the effect before rerunning:

- In `synth.py`, set `android_shown_control=0.03` for streak_badge. Is the SRM still caught?
- Set `drift_sd=0.0`. What happens to CUPED's variance reduction, and why?
- In `simulate.peeking`, change `days` to 60. What happens to the t-test false-positive rate?
  And to mSPRT?

## 2. Read the code in this order

| File | What to understand | Question to answer yourself |
| --- | --- | --- |
| `experiments.yml`, `metrics.yml` | what an experiment and a metric are | Why is the primary metric chosen before the test? |
| `config.py` | validation at load time | Why reject column names that are not identifiers? |
| `assignment.py` | two hashes, buckets, layers | Why a second salt for the variant? What breaks if I edit a salt mid-test? |
| `synth.py` | `lam`, drift, Poisson, the five plants | Why does pre-period activity predict in-test activity? |
| `sql/user_metrics.sql` | first exposure, window, LEFT JOIN | Why keep users with zero events? Why exclude users whose window has not ended? |
| `sql/ratio_metric.sql` | delta-method variance | Why not a binomial variance over lessons? |
| `checks.py` | SRM, balance, MDE | Why p < 0.001 for SRM and not 0.05? |
| `stats.py` | Welch, CUPED, Holm | Derive theta = cov/var. Why is CUPED unbiased? |
| `sequential.py` | Lambda, confidence sequence | Why can you look every day with mSPRT? (Ville's inequality) |
| `analysis.py` | the decision order | Why do checks come before effects? Why does a guardrail need both conditions? |
| `simulate.py` | vectorised Monte Carlo | Why cumulative sums for the daily looks? |
| `report.py` | numbers come from code | Where does each number in the README come from? |

## 3. Key concepts in my own words (fill these in)

- **Randomisation unit vs analysis unit:** ...
- **Exposure vs assignment:** ...
- **Sample ratio mismatch:** ...
- **p-value and confidence interval (what they do and do not say):** ...
- **Power and minimum detectable effect:** ...
- **CUPED / regression adjustment:** ...
- **Delta method:** ...
- **Peeking and optional stopping:** ...
- **mSPRT, always-valid p-values, confidence sequences:** ...
- **Family-wise error rate and Holm:** ...
- **Guardrail metrics:** ...

## 4. Interview questions this repo answers

1. *How do you assign users to variants?* Hash salt + user id; buckets give traffic allocation;
   layers give mutual exclusion; a second salt gives the split. Stateless and sticky.
2. *Your experiment shows a 49/51 split with 100,000 users. Problem?* Chi-square SRM: with
   100,000 users, 49,000 vs 51,000 gives chi-square 40, p about 2.5e-10. Check by platform, app version,
   country; streak_badge is the worked example.
3. *How do you reduce variance?* CUPED: theta = cov(Y, X)/var(X) with a pre-period X; variance
   falls by corr^2; here about 34%, CI 19% narrower. Unbiased because X is pre-treatment.
4. *How do you test a ratio metric like completion rate?* Delta method on per-user sums, because
   the user is the randomisation unit. Show the formula and the bootstrap test.
5. *Why can't we stop when p < 0.05?* Show the peeking chart: 5.6% after one look, 28.7% after
   28. Fix: fixed horizon with no peeking, group-sequential with a planned schedule, or mSPRT.
6. *mSPRT vs O'Brien-Fleming?* Any-time validity vs more power for a fixed look schedule.
7. *The primary metric won but crashes went up. Ship?* No: guardrail logic, and the business
   reason (crashes cost trust and reviews; one week of lessons does not pay for that).
8. *How many users do we need?* `n = 2 var (z_0.975 + z_0.8)^2 / effect^2`; 15,577 per arm for
   +3% here, fewer with CUPED.
9. *How do you know your statistics code is right?* Simulations with known truth: A/A
   false-positive rate 4.5% (3.9% to 5.2%), CUPED unbiased, peeking inflation measured.

## 5. Make it mine (do at least three)

- [ ] Export a week of real HaiEduTech logs in the three-table schema and run `xp analyze`.
- [ ] Add a novelty effect to the generator (lift that fades over days since exposure) and a
      report section that plots the effect by day since exposure.
- [ ] Implement O'Brien-Fleming alpha spending for 4 planned looks and compare its power with
      mSPRT in `simulate.py`.
- [ ] Add a non-inferiority test for guardrails with its own power calculation.
- [ ] Add a "segment" table (platform, new vs returning) with Holm correction, and explain why
      it must not drive the decision.
- [ ] Write a LinkedIn post: "Why I don't trust a green dashboard on day 3", with the chart.

## 6. Quality checks I should be able to explain

- `make check` = `ruff check`, `ruff format --check`, `mypy src`, `pytest --cov-fail-under=80`.
- Why the hash test pins exact values (a library change must not move users between variants).
- Why the SQL test uses a hand-made dataset (each number can be checked on paper).
- Why the decision test runs on the same seeded logs as `docs/results.md`.

## 7. Honesty checklist before pushing

- [ ] I can explain every file without looking at it.
- [ ] Every number in the README is from `docs/results.md`, which I regenerated myself.
- [ ] The README says the data is synthetic, the effects were planted, and the seed was chosen
      to show the peeking trap.
- [ ] I never say "the reminder increases lessons by 3%" about a real app.
- [ ] The FastAPI service is described as a demo (no auth, no rate limits, exposure log is a file).
- [ ] Anything I have not built is under "What I'd do next".
