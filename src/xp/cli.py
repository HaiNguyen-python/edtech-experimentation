"""Command line: xp generate | analyze | report | assign | serve."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("xp")
DATA = Path("data")


def cmd_generate(args: argparse.Namespace) -> None:
    from . import synth

    tables = synth.generate(args.seed, args.scale)
    synth.write(tables, args.out)
    sizes = {name: len(df) for name, df in tables.items()}
    log.info("wrote %s to %s", sizes, args.out)


def cmd_analyze(args: argparse.Namespace) -> None:
    from . import pipeline, synth
    from .analysis import analyze
    from .config import load_experiments, load_metrics
    from .report import write_experiment

    experiments = load_experiments()
    if args.experiment not in experiments:
        raise SystemExit(f"unknown experiment {args.experiment}; known: {sorted(experiments)}")
    if not (args.data / "events.parquet").exists():
        log.info("no data in %s: generating the synthetic logs first", args.data)
        synth.write(synth.generate(), args.data)
    con = pipeline.connect(args.data)
    result = analyze(con, experiments[args.experiment], load_metrics())
    path = write_experiment(result, args.out)
    print(f"{args.experiment}: {result.decision.upper()}")
    for reason in result.reasons:
        print(f"  - {reason}")
    log.info("wrote %s", path)


def cmd_report(args: argparse.Namespace) -> None:
    from .report import StudyConfig, summary_frame, write

    study = write(args.out, StudyConfig(), args.data)
    print(summary_frame(study).to_string(index=False))
    log.info("wrote %s, experiment reports and charts", args.out / "results.md")


def cmd_assign(args: argparse.Namespace) -> None:
    from .assignment import Assigner

    print(json.dumps(Assigner.from_catalog().assign_all(args.user_id), indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(args.exposure_log), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write seeded synthetic logs as Parquet")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--scale", type=float, default=1.0, help="1.0 = 65,000 users")
    p.add_argument("--out", type=Path, default=DATA)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("analyze", help="checks, statistics and a report for one experiment")
    p.add_argument("experiment")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="all experiments + simulations -> docs/results.md")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("assign", help="show a user's variant in every experiment")
    p.add_argument("user_id")
    p.set_defaults(func=cmd_assign)

    p = sub.add_parser("serve", help="FastAPI /assign endpoint (needs the [api] extra)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--exposure-log", type=Path, default=None)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if getattr(args, "seed", 0) is None:
        from .synth import DEFAULT_SEED

        args.seed = DEFAULT_SEED
    args.func(args)


if __name__ == "__main__":
    main()
