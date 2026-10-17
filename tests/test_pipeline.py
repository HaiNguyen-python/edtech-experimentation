from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from conftest import make_experiment

from xp import pipeline
from xp.config import load_metrics


def test_user_metrics_on_hand_made_logs(tiny_logs):
    con = pipeline.connect(tiny_logs)
    um = pipeline.build_user_metrics(con, make_experiment(), as_of=datetime(2026, 1, 25)).set_index(
        "user_id"
    )
    assert list(um.index) == ["a", "b", "c"]  # d: other experiment, e: window not ended
    a, b, c = um.loc["a"], um.loc["b"], um.loc["c"]
    assert a["variant"] == "control"  # first exposure wins, duplicates ignored
    assert (a["lessons_completed"], a["lessons_started"], a["active_days"]) == (2, 3, 2)
    assert a["lessons_completed_per_week"] == pytest.approx(2.0)
    assert a["pre_lessons_completed_per_week"] == pytest.approx(0.5)  # 1 lesson in 2 weeks
    assert (a["crashed"], a["unsubscribed"]) == (1, 0)
    assert b[["lessons_completed", "lessons_started", "active_days", "crashed"]].sum() == 0
    assert (c["lessons_completed"], c["active_days"], c["unsubscribed"]) == (2, 1, 1)


def test_summaries_match_pandas(tiny_logs):
    con = pipeline.connect(tiny_logs)
    um = pipeline.build_user_metrics(con, make_experiment(), as_of=datetime(2026, 1, 25))
    s = pipeline.mean_summary(con)
    row = s[(s["metric"] == "lessons_completed") & (s["variant"] == "treatment")].iloc[0]
    expected = um.loc[um["variant"] == "treatment", "lessons_completed"]
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(expected.mean())
    assert row["variance"] == pytest.approx(expected.var(ddof=1))
    counts = pipeline.assignment_counts(con)
    assert counts["users"].sum() == 3


def test_delta_method_variance_matches_bootstrap():
    rng = np.random.default_rng(0)
    n = 400
    starts = rng.poisson(rng.gamma(2.0, 2.0, 2 * n)) + 1
    done = rng.binomial(starts, rng.beta(8, 2, 2 * n))
    users = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(2 * n)],
            "variant": ["control"] * n + ["treatment"] * n,
            "lessons_completed": done,
            "lessons_started": starts,
        }
    )
    con = pipeline.connect(
        {
            "users": pd.DataFrame({"user_id": [], "platform": []}),
            "exposures": pd.DataFrame(
                {"experiment": [], "user_id": [], "variant": [], "exposed_at": []}
            ),
            "events": pd.DataFrame({"user_id": [], "event_time": [], "event_name": []}),
        }
    )
    con.register("um_df", users)
    con.execute("CREATE TEMP TABLE user_metrics AS SELECT * FROM um_df")
    sql = pipeline.ratio_summary(con, load_metrics()["completion_rate"]).set_index("variant")
    ctrl = users[users["variant"] == "control"]
    x, y = ctrl["lessons_completed"].to_numpy(), ctrl["lessons_started"].to_numpy()
    assert sql.loc["control", "ratio"] == pytest.approx(x.sum() / y.sum())
    idx = rng.integers(0, n, (4000, n))
    boot = x[idx].sum(axis=1) / y[idx].sum(axis=1)
    assert sql.loc["control", "ratio_variance"] == pytest.approx(boot.var(), rel=0.12)


def test_connect_reads_files_and_rejects_missing_columns(tiny_logs, tmp_path):
    for name, df in tiny_logs.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    con = pipeline.connect(tmp_path)
    assert con.execute("SELECT count(*) FROM events").fetchone() == (16,)
    broken = dict(tiny_logs, users=tiny_logs["users"].drop(columns="platform"))
    with pytest.raises(ValueError, match="users is missing"):
        pipeline.connect(broken)
    with pytest.raises(FileNotFoundError):
        pipeline.connect(tmp_path / "nowhere")


def test_daily_looks_only_count_users_whose_window_ended(tiny_logs):
    con = pipeline.connect(tiny_logs)
    exp = make_experiment()
    pipeline.build_user_metrics(con, exp, as_of=datetime(2026, 1, 25))
    looks = pipeline.daily_looks(con, exp, "lessons_completed_per_week", 7, 9)
    per_day = looks.groupby("day")["n"].sum().to_dict()
    # a ends 2026-01-17 08:00 (day 8 is 01-18 00:00); b and c end on 01-18 (day 8 / day 9)
    assert per_day == {8: 2, 9: 3}
    with pytest.raises(ValueError, match="identifier"):
        pipeline.daily_looks(con, exp, "x) OR (1=1", 7, 9)
