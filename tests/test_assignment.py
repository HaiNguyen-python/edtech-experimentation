import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from xp.assignment import Assigner, ExposureLogger, assign, bucket, hash_unit, pick_variant
from xp.config import load_experiments, load_layers, load_metrics

USERS = [f"u{i:06d}" for i in range(20_000)]


def test_hash_is_stable_across_runs_and_versions():
    # Fixed values: if these change, every user would silently switch variants.
    assert bucket("layer-home-v1", "u000001") == 436
    assert bucket("salt", "user-42") == 661
    assert hash_unit("practice-reminder-2026-03", "u000001") == pytest.approx(0.74725974978)


def test_different_salts_give_unrelated_buckets():
    a = np.array([bucket("salt-a", u) for u in USERS[:5000]])
    b = np.array([bucket("salt-b", u) for u in USERS[:5000]])
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.05


def test_buckets_are_uniform_and_variants_split_by_weight():
    counts = np.bincount([bucket("layer-x", u, 20) for u in USERS], minlength=20)
    assert stats.chisquare(counts).pvalue > 0.001
    picks = [pick_variant({"a": 0.2, "b": 0.8}, hash_unit("exp-x", u)) for u in USERS]
    share_a = picks.count("a") / len(picks)
    assert abs(share_a - 0.2) < 0.01  # 3.5 standard errors at n = 20,000


def test_layers_give_mutual_exclusion_and_traffic_allocation():
    exps, layers = load_experiments(), load_layers()
    home = [e for e in exps.values() if e.layer == "home"]
    in_home = {e.name: {u for u in USERS if assign(e, layers["home"], u)} for e in home}
    names = list(in_home)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not in_home[a] & in_home[b], f"{a} and {b} share users"
    # streak_badge owns buckets 200-599: 40% of the layer's traffic
    assert abs(len(in_home["streak_badge"]) / len(USERS) - 0.4) < 0.015


def test_different_layers_are_independent():
    exps, layers = load_experiments(), load_layers()
    reminder, streak = exps["practice_reminder"], exps["streak_badge"]
    streak_users = [u for u in USERS if assign(streak, layers["home"], u)]
    treated = [assign(reminder, layers["notifications"], u) == "treatment" for u in streak_users]
    assert abs(np.mean(treated) - 0.5) < 0.03


def test_assigner_and_exposure_logger(tmp_path: Path):
    assigner = Assigner.from_catalog()
    out = assigner.assign_all("u000123")
    assert out == assigner.assign_all("u000123")
    assert sum(load_experiments()[name].layer == "home" for name in out) <= 1
    logger = ExposureLogger(tmp_path / "exposures.jsonl")
    logger.log("practice_reminder", "u000123", out["practice_reminder"])
    record = json.loads((tmp_path / "exposures.jsonl").read_text().splitlines()[0])
    assert record["user_id"] == "u000123" and record["experiment"] == "practice_reminder"


def test_catalog_validation_rejects_overlaps_and_unsafe_columns(tmp_path: Path):
    bad = tmp_path / "experiments.yml"
    text = (Path(__file__).parents[1] / "src/xp/experiments.yml").read_text()
    bad.write_text(text.replace("buckets: [200, 600]", "buckets: [100, 600]"))
    with pytest.raises(ValueError, match="overlap"):
        load_experiments(bad)
    metrics = tmp_path / "metrics.yml"
    metrics.write_text(
        "metrics:\n  - {name: m, description: x, type: mean, direction: increase,"
        " column: 'x; DROP TABLE events'}\n"
    )
    with pytest.raises(ValueError, match="bad column"):
        load_metrics(metrics)
