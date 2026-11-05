"""FastAPI app.

    uvicorn rag.api:app --host 0.0.0.0 --port 8000

The traffic simulator can tag requests with ``X-Trace-Env: sim`` and backdate them with
``X-Trace-Timestamp``, but only when the server has ``SIM_SECRET`` set and the request carries the
same value in ``X-Sim-Secret``. Otherwise those headers are ignored.
"""
from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from rag.service import AskResponse, RAGService

logger = logging.getLogger(__name__)
_service: RAGService | None = None


def service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService.build()
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    service()  # warm up corpus, indexes and guard models before taking traffic
    yield
    if _service is not None:
        _service.c.tracer.flush()


app = FastAPI(title="SEC 10-K RAG service with guardrails", version="0.2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    tickers: list[str] | None = None
    include_guards: bool = True


@app.get("/health")
def health() -> dict:
    s = service()
    return {"status": "ok", "chunks": len(s.c.corpus), "embedder": s.c.embedder.name,
            "provider": s.c.generator.provider, "guards_config": s.c.pipeline.version}


@app.get("/config")
def config() -> dict:
    s = service()
    return {"version": s.c.pipeline.version, "modes": s.c.pipeline.modes(), "companies": s.companies}


def _sim_overrides(env: str | None, ts: str | None, secret: str | None) -> tuple[str | None, datetime | None]:
    expected = os.environ.get("SIM_SECRET", "")
    if not (env or ts):
        return None, None
    if not expected or not secret or not hmac.compare_digest(expected, secret):
        return None, None
    stamp = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
    return env, stamp


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, background: BackgroundTasks,
        x_trace_env: str | None = Header(default=None), x_trace_timestamp: str | None = Header(default=None),
        x_sim_secret: str | None = Header(default=None)) -> AskResponse:
    s = service()
    try:
        env, stamp = _sim_overrides(x_trace_env, x_trace_timestamp, x_sim_secret)
    except ValueError as e:
        raise HTTPException(400, f"bad X-Trace-Timestamp: {e}") from e
    resp = s.ask(req.question, req.tickers, environment=env, timestamp=stamp, trace="defer")
    if resp.trace_record is not None:
        background.add_task(s.c.tracer.log, resp.trace_record)
    if not req.include_guards:
        resp.guards = []
    return resp


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=8, max_length=64)
    thumbs_up: bool
    comment: str | None = Field(default=None, max_length=1000)


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Thumbs up/down on an answer; thumbs-down traces enter the review queue (monitor/feedback.py)."""
    s = service()
    backend = s.c.tracer.backend
    if not hasattr(backend, "write_feedback"):
        raise HTTPException(501, "tracing backend does not store feedback")
    comment = s.c.pipeline.mask(req.comment) if req.comment else None
    backend.write_feedback(req.trace_id, 1.0 if req.thumbs_up else 0.0, s.c.tracer.environment, comment)
    return {"ok": True}
