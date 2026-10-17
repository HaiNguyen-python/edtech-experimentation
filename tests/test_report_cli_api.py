import json

import pytest
from fastapi.testclient import TestClient

from xp import pipeline
from xp.analysis import analyze
from xp.api import create_app
from xp.cli import main
from xp.config import load_experiments, load_metrics
from xp.report import StudyConfig, experiment_markdown, results_markdown, write

TINY_STUDY = StudyConfig(
    aa_reps=200,
    lift_reps=100,
    n_per_arm=300,
    peeking_reps=100,
    power_reps=50,
    power_sizes=(500, 2000),
)


def test_experiment_report_has_decision_checks_metrics_and_chart(study_data):
    con = pipeline.connect(study_data)
    result = analyze(con, load_experiments()["new_onboarding"], load_metrics())
    md = experiment_markdown(result, "chart.png")
    for text in (
        "## Decision: **DON'T SHIP**",
        "Sample ratio (SRM)",
        "Pre-period balance",
        "Power",
        "| crash_rate | guardrail |",
        "fail",
        "![new_onboarding](chart.png)",
        "Synthetic data",
    ):
        assert text in md


def test_full_report_writes_results_and_charts(study_data, tmp_path):
    study = write(tmp_path, TINY_STUDY, data_dir=study_data)
    md = (tmp_path / "results.md").read_text()
    assert "Peeking every day" in md and "CUPED is unbiased" in md and "Synthetic data" in md
    assert len(study.results) == 5
    for name in ("peeking_false_positives", "reminder_effect"):
        png = tmp_path / "img" / f"{name}.png"
        assert png.exists() and png.stat().st_size < 300_000
    assert (tmp_path / "experiments" / "home_redesign.md").exists()
    assert results_markdown(study) == md  # deterministic given the same study


def test_cli_generate_analyze_and_assign(tmp_path, capsys):
    data = tmp_path / "data"
    main(["generate", "--scale", "0.05", "--out", str(data)])
    assert (data / "events.parquet").exists()
    main(["analyze", "practice_reminder", "--data", str(data), "--out", str(tmp_path / "r")])
    assert (tmp_path / "r" / "practice_reminder.md").exists()
    main(["assign", "u000001"])
    out = capsys.readouterr().out
    assert "practice_reminder:" in out and '"practice_reminder"' in out
    with pytest.raises(SystemExit):
        main(["analyze", "nope", "--data", str(data)])


def test_api_assigns_deterministically_and_logs_only_reached_experiments(tmp_path):
    log = tmp_path / "exp.jsonl"
    client = TestClient(create_app(log))
    assert client.get("/health").json() == {"status": "ok"}
    body = {"user_id": "u000001", "exposed": ["practice_reminder"]}
    first = client.post("/assign", json=body).json()
    assert first == client.post("/assign", json=body).json()
    assert first["logged_exposures"] == ["practice_reminder"]
    assert len(log.read_text().splitlines()) == 2
    assert json.loads(log.read_text().splitlines()[0])["variant"] in ("control", "treatment")
    assert client.post("/assign", json={"user_id": "u1", "exposed": ["nope"]}).status_code == 422
