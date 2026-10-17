"""Optional live assignment service: POST /assign returns a user's variant per experiment.

Stateless: the variant is a pure function of (salt, user id), so any number of replicas give
the same answer without a shared database. Exposures are logged only for the experiments the
client says the user actually reached (`exposed`), matching the analysis, which counts users
from their first exposure.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .assignment import Assigner, ExposureLogger


class AssignRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    exposed: list[str] = Field(default_factory=list, description="experiments the user reached")


class AssignResponse(BaseModel):
    user_id: str
    assignments: dict[str, str]
    logged_exposures: list[str]


def create_app(exposure_log: Path | None = None, assigner: Assigner | None = None) -> FastAPI:
    app = FastAPI(title="xp assignment", version="0.1.0")
    assigner = assigner or Assigner.from_catalog()
    logger = ExposureLogger(exposure_log)
    app.state.exposures = logger

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/assign", response_model=AssignResponse)
    def assign(req: AssignRequest) -> AssignResponse:
        assignments = assigner.assign_all(req.user_id)
        unknown = [e for e in req.exposed if e not in assigner.experiments]
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown experiments: {unknown}")
        logged = [e for e in req.exposed if e in assignments]
        for name in logged:
            logger.log(name, req.user_id, assignments[name])
        return AssignResponse(user_id=req.user_id, assignments=assignments, logged_exposures=logged)

    return app
