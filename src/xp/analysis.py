"""Analyse one experiment end to end and make a decision: ship, don't ship, or investigate.

Order matters and mirrors how a careful analyst works:
1. Build per-user metrics in SQL (exposure-filtered).
2. Checks: SRM, pre-period balance, power. A failed SRM or balance check stops the analysis
   at "investigate": the effect estimate is not trustworthy, so it is shown but not used.
3. Primary metric: Welch t-test, plus CUPED when a pre-period covariate exists. Monitored
   experiments are judged on the always-valid mSPRT p-value instead.
4. Secondary metrics with Holm-adjusted p-values (they inform, they do not decide).
5. Guardrails: any significant harm beyond the catalog threshold blocks shipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from . import pipeline
from .checks import BalanceCheck, PowerCheck, SrmCheck, balance_check, power_check, srm_check
from .config import Experiment, Metric
from .sequential import monitor
from .stats import ALPHA, CupedResult, Estimate, cuped, holm, ratio_difference, welch

SHIP, DONT_SHIP, INVESTIGATE = "ship", "don't ship", "investigate"


@dataclass(frozen=True)
class MetricRow:
    metric: Metric
    role: str  # primary, secondary, guardrail
    method: str
    estimate: Estimate
    p_adjusted: float | None = None
    status: str = ""


@dataclass
class ExperimentResult:
    experiment: Experiment
    n_users: int
    srm: SrmCheck
    balance: BalanceCheck
    power: PowerCheck
    primary_plain: Estimate
    primary_cuped: CupedResult | None
    rows: list[MetricRow]
    monitoring: pd.DataFrame | None
    decision: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def primary(self) -> Estimate:
        """The estimate the decision uses: CUPED when available (same target, less noise)."""
        return self.primary_cuped.estimate if self.primary_cuped else self.primary_plain

    def guardrails(self) -> list[MetricRow]:
        return [r for r in self.rows if r.role == "guardrail"]

    def naive_win_day(self) -> int | None:
        """First look day on which a daily t-test showed a significant WIN (the peeking trap)."""
        if self.monitoring is None:
            return None
        mon = self.monitoring
        sign = 1.0 if self.rows[0].metric.direction == "increase" else -1.0
        hits = mon.loc[(mon["naive_p"] < ALPHA) & (sign * mon["diff"] > 0), "day"]
        return int(hits.iloc[0]) if len(hits) else None


def _mean_estimate(summary: pd.DataFrame, column: str, exp: Experiment) -> Estimate:
    s = summary[summary["metric"] == column].set_index("variant").to_dict("index")
    t, c = s[exp.treatment], s[exp.control]
    return welch(
        float(t["mean"]),
        float(t["variance"]),
        int(t["n"]),
        float(c["mean"]),
        float(c["variance"]),
        int(c["n"]),
    )


def metric_estimate(
    con: duckdb.DuckDBPyConnection, summary: pd.DataFrame, metric: Metric, exp: Experiment
) -> tuple[str, Estimate]:
    if metric.type == "ratio":
        r = pipeline.ratio_summary(con, metric).set_index("variant").to_dict("index")
        t, c = r[exp.treatment], r[exp.control]
        est = ratio_difference(
            float(t["ratio"]),
            float(t["ratio_variance"]),
            float(c["ratio"]),
            float(c["ratio_variance"]),
        )
        return "delta method", est
    assert metric.column is not None
    return "Welch t-test", _mean_estimate(summary, metric.column, exp)


def effect_status(est: Estimate, metric: Metric) -> str:
    if not est.significant:
        return "flat"
    good = est.diff > 0 if metric.direction == "increase" else est.diff < 0
    return "win" if good else "loss"


def guardrail_status(est: Estimate, metric: Metric) -> str:
    """fail = significantly worse AND worse than the threshold; warn = only one of the two.

    Requiring both for "fail" avoids blocking a launch over a tiny, real but harmless change,
    and avoids blocking it over a large but pure-noise point estimate.
    """
    threshold = metric.guardrail_threshold or 0.0
    harm_sign = 1.0 if metric.direction == "decrease" else -1.0
    significant_harm = est.ci_low > 0 if harm_sign > 0 else est.ci_high < 0
    over = harm_sign * est.rel > threshold
    if significant_harm and over:
        return "fail"
    if significant_harm or over:
        return "warn"
    return "pass"


def decide(result: ExperimentResult) -> tuple[str, list[str]]:
    exp, reasons = result.experiment, []
    if result.srm.flagged:
        worst = result.srm.by_platform.sort_values("p_value").iloc[0]
        reasons.append(
            f"Sample ratio mismatch: p = {result.srm.p_value:.1e} (threshold 0.001). "
            f"Worst platform: {worst['platform']} with {worst['share_treatment']:.1%} of users "
            "in treatment instead of 50%. The arms are not comparable; fix assignment or "
            "logging and rerun."
        )
    if result.balance.flagged:
        reasons.append("Pre-period balance check failed: the arms differed before the change.")
    if reasons:
        return INVESTIGATE, reasons

    primary = result.primary
    if exp.analysis == "sequential" and result.monitoring is not None:
        last = result.monitoring.iloc[-1]
        p_seq = float(last["always_valid_p"])
        win_day = result.naive_win_day()
        if win_day is not None:
            reasons.append(
                f"A daily t-test showed a significant win on day {win_day}; a t-test is not "
                "valid when you look every day."
            )
        if p_seq >= ALPHA:
            peek_note = (
                " The early win is what peeking produces under no effect." if win_day else ""
            )
            reasons.append(
                f"Always-valid (mSPRT) p = {p_seq:.2f} after {int(last['day'])} days: "
                f"no reliable effect.{peek_note}"
            )
            return DONT_SHIP, reasons
        reasons.append(f"Always-valid (mSPRT) p = {p_seq:.4f}: the effect is reliable.")

    if exp.aa:
        secondary_hits = [
            r for r in result.rows if r.role == "secondary" and (r.p_adjusted or 1) < ALPHA
        ]
        if primary.significant or secondary_hits:
            return INVESTIGATE, [
                *reasons,
                "A/A test shows a significant difference; this happens 5% of the time by "
                "chance, so rerun before suspecting the platform.",
            ]
        return DONT_SHIP, [
            *reasons,
            "A/A test: no difference expected and none found. Assignment, logging and "
            "statistics look healthy (nothing to ship).",
        ]

    failed = [r for r in result.guardrails() if r.status == "fail"]
    for r in failed:
        reasons.append(
            f"Guardrail {r.metric.name} is significantly worse: {r.estimate.rel:+.1%} "
            f"(95% CI {r.estimate.rel_ci_low:+.1%} to {r.estimate.rel_ci_high:+.1%}), above "
            f"the {r.metric.guardrail_threshold:.0%} limit."
        )
    for r in result.guardrails():
        if r.status == "warn":
            reasons.append(
                f"Watch {r.metric.name}: {r.estimate.rel:+.1%} (not a blocker: either not "
                "significant or below the threshold)."
            )
    if failed:
        return DONT_SHIP, reasons

    method = "CUPED" if result.primary_cuped else "Welch t-test"
    status = effect_status(primary, result.rows[0].metric)
    text = (
        f"Primary metric {exp.primary}: {primary.rel:+.2%} "
        f"(95% CI {primary.rel_ci_low:+.2%} to {primary.rel_ci_high:+.2%}, {method})."
    )
    if status == "win":
        return SHIP, [text, *reasons]
    if status == "loss":
        return DONT_SHIP, [text + " The change makes it worse.", *reasons]
    return DONT_SHIP, [
        text + f" No detectable effect (MDE was {result.power.mde_rel:.1%}).",
        *reasons,
    ]


def analyze(
    con: duckdb.DuckDBPyConnection, exp: Experiment, metrics: dict[str, Metric]
) -> ExperimentResult:
    users = pipeline.build_user_metrics(con, exp)
    summary = pipeline.mean_summary(con)
    primary = metrics[exp.primary]
    assert primary.column is not None, "the primary metric must be a mean metric"

    treated = (users["variant"] == exp.treatment).to_numpy()
    y = users[primary.column].to_numpy(dtype=float)
    cuped_res = None
    if primary.covariate is not None and not exp.new_users:
        cuped_res = cuped(y, users[primary.covariate].to_numpy(dtype=float), treated)

    plain = _mean_estimate(summary, primary.column, exp)
    main = cuped_res.estimate if cuped_res else plain
    rows = [
        MetricRow(
            primary,
            "primary",
            "CUPED" if cuped_res else "Welch t-test",
            main,
            status=effect_status(main, primary),
        )
    ]
    secondary = [metrics[n] for n in exp.secondary]
    estimates = [metric_estimate(con, summary, m, exp) for m in secondary]
    adjusted = holm([est.p_value for _, est in estimates])
    for metric, (method, est), p_adj in zip(secondary, estimates, adjusted, strict=True):
        rows.append(MetricRow(metric, "secondary", method, est, p_adj, effect_status(est, metric)))
    for name in exp.guardrails:
        metric = metrics[name]
        method, est = metric_estimate(con, summary, metric, exp)
        rows.append(
            MetricRow(metric, "guardrail", method, est, None, guardrail_status(est, metric))
        )

    monitoring = None
    if exp.analysis == "sequential":
        looks = pipeline.daily_looks(
            con, exp, primary.column, exp.window_days + 1, exp.ramp_days + exp.window_days
        )
        baseline = float(np.mean(y[~treated]))
        monitoring = monitor(looks, exp.control, exp.treatment, tau=exp.planned_mde * baseline)

    result = ExperimentResult(
        experiment=exp,
        n_users=len(users),
        srm=srm_check(pipeline.assignment_counts(con), exp),
        balance=balance_check(users, exp, None if exp.new_users else primary.covariate),
        power=power_check(
            users, exp, primary.column, cuped_res.variance_reduction if cuped_res else None
        ),
        primary_plain=plain,
        primary_cuped=cuped_res,
        rows=rows,
        monitoring=monitoring,
    )
    result.decision, result.reasons = decide(result)
    return result
