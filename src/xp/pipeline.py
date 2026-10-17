"""The metrics pipeline: logs in, per-user and per-variant metrics out, all in DuckDB SQL.

Why SQL: in a real company the logs live in a warehouse, and metric logic must run there, close
to the data, and be readable by analysts. DuckDB runs the same SQL in-process, so the pipeline
is testable on a laptop and in CI. The SQL files are in `src/xp/sql/`; Python only binds
parameters and reads small summary tables back.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

from .config import IDENTIFIER, Experiment, Metric

PRE_DAYS = 14
SCHEMA: dict[str, set[str]] = {
    "users": {"user_id", "platform"},
    "exposures": {"experiment", "user_id", "variant", "exposed_at"},
    "events": {"user_id", "event_time", "event_name"},
}


def read_sql(name: str) -> str:
    return resources.files("xp").joinpath("sql", f"{name}.sql").read_text()


def _ident(name: str) -> str:
    """Only plain lower-case identifiers may be formatted into SQL (no injection via YAML)."""
    if not IDENTIFIER.match(name):
        raise ValueError(f"not a safe SQL identifier: {name!r}")
    return name


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def connect(source: Path | Mapping[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with views `users`, `exposures`, `events`.

    `source` is either a folder with <table>.parquet or <table>.csv files (the generator's
    output, or real logs exported in the same schema), or a dict of DataFrames (tests).
    Missing columns fail here, with the table name, not deep inside a query.
    """
    con = duckdb.connect()
    for table, required in SCHEMA.items():
        if isinstance(source, Mapping):
            con.register(f"{table}_df", source[table])
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM {table}_df")
        else:
            parquet, csv = source / f"{table}.parquet", source / f"{table}.csv"
            # Views cannot take bound parameters, so the path is quoted as a SQL string literal.
            if parquet.exists():
                con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet({_quote(parquet)})")
            elif csv.exists():
                con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto({_quote(csv)})")
            else:
                raise FileNotFoundError(f"{table}.parquet or {table}.csv not found in {source}")
        cols = {row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()}
        missing = required - cols
        if missing:
            raise ValueError(f"table {table} is missing columns {sorted(missing)}")
    return con


def build_user_metrics(
    con: duckdb.DuckDBPyConnection,
    exp: Experiment,
    as_of: datetime | None = None,
    pre_days: int = PRE_DAYS,
) -> pd.DataFrame:
    """Create the temp table `user_metrics` for one experiment and return it."""
    params = {
        "experiment": exp.name,
        "window_days": exp.window_days,
        "pre_days": pre_days,
        "as_of": as_of or exp.end + timedelta(days=1),
    }
    con.execute(f"CREATE OR REPLACE TEMP TABLE user_metrics AS {read_sql('user_metrics')}", params)
    return con.execute("SELECT * FROM user_metrics").df()


def mean_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """(variant, metric) -> n, mean, variance for every numeric per-user column."""
    return con.execute(read_sql("mean_metrics")).df()


def ratio_summary(con: duckdb.DuckDBPyConnection, metric: Metric) -> pd.DataFrame:
    """variant -> n, ratio, delta-method variance of the ratio."""
    if metric.numerator is None or metric.denominator is None:
        raise ValueError(f"{metric.name} is not a ratio metric")
    sql = read_sql("ratio_metric").format(
        numerator=_ident(metric.numerator), denominator=_ident(metric.denominator)
    )
    return con.execute(sql).df()


def assignment_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(read_sql("assignment_counts")).df()


def daily_looks(
    con: duckdb.DuckDBPyConnection, exp: Experiment, column: str, first_day: int, last_day: int
) -> pd.DataFrame:
    """(day, variant) -> n, mean, variance of `column` as seen on each look day."""
    sql = read_sql("daily_looks").format(metric=_ident(column))
    params = {
        "start": exp.start,
        "window_days": exp.window_days,
        "first_day": first_day,
        "last_day": last_day,
    }
    return con.execute(sql, params).df()
