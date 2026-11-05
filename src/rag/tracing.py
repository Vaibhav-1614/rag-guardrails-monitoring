"""Request tracing with PII masking.

Backends:
- ``langfuse``: Langfuse public REST API (batched ``/api/public/ingestion`` writes, ``/api/public/traces``
  reads). Guard results go into trace metadata and a few numeric scores, not extra spans, to stay
  inside the Hobby tier's unit budget.
- ``local``: JSONL files under ``data/traces/{environment}/{YYYY-MM-DD}.jsonl``. Used when no Langfuse
  keys are configured, and by the traffic simulator / tests.

Every string in a record passes through ``mask`` (the PII redactor) before it reaches any backend,
so raw PII never leaves the process, whichever guard modes are configured.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from rag.config import load_settings, resolve

logger = logging.getLogger(__name__)

# Metadata keys whose string values are machine-generated (ids, enums) and never need masking.
_NO_MASK_KEYS = {"guard", "stage", "action", "mode", "chunk_id", "ticker", "section", "provider", "model", "status",
                 "environment", "embedder", "strategy", "config_version", "refusal_reason", "id", "rule", "entity_type"}


class TraceRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    environment: str = "production"
    name: str = "rag.ask"
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


def mask_value(value: Any, mask: Callable[[str], str], key: str | None = None) -> Any:
    if isinstance(value, str):
        return value if key in _NO_MASK_KEYS or len(value) < 6 else mask(value)
    if isinstance(value, dict):
        return {k: mask_value(v, mask, k) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [mask_value(v, mask, key) for v in value]
    return value


class TraceBackend(Protocol):
    name: str

    def write(self, record: TraceRecord) -> None: ...

    def read(self, start: datetime, end: datetime, environment: str | None = None) -> list[dict]: ...

    def flush(self) -> None: ...


class NullBackend:
    name = "none"

    def write(self, record: TraceRecord) -> None:
        pass

    def read(self, start: datetime, end: datetime, environment: str | None = None) -> list[dict]:
        return []

    def flush(self) -> None:
        pass


class LocalBackend:
    name = "local"

    def __init__(self, directory: str | Path) -> None:
        self.dir = resolve(directory)
        self._lock = threading.Lock()

    def _path(self, env: str, ts: datetime) -> Path:
        return self.dir / env / f"{ts.astimezone(UTC):%Y-%m-%d}.jsonl"

    def write(self, record: TraceRecord) -> None:
        p = self._path(record.environment, record.timestamp)
        line = record.model_dump_json()
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read(self, start: datetime, end: datetime, environment: str | None = None) -> list[dict]:
        envs = [environment] if environment else [d.name for d in self.dir.glob("*") if d.is_dir()]
        out: list[dict] = []
        day = start.astimezone(UTC).date()
        while day <= end.astimezone(UTC).date():
            for env in envs:
                p = self.dir / env / f"{day:%Y-%m-%d}.jsonl"
                if not p.exists():
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    ts = datetime.fromisoformat(rec["timestamp"].replace("Z", "+00:00"))
                    if start <= ts < end:
                        out.append(rec)
            day += timedelta(days=1)
        return sorted(out, key=lambda r: r["timestamp"])

    def write_feedback(self, trace_id: str, value: float, environment: str, comment: str | None = None) -> None:
        p = self.dir / environment / "feedback.jsonl"
        row = {"trace_id": trace_id, "value": value, "comment": comment, "timestamp": datetime.now(UTC).isoformat()}
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def read_feedback(self, start: datetime, end: datetime, environment: str) -> dict[str, float]:
        p = self.dir / environment / "feedback.jsonl"
        if not p.exists():
            return {}
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if start <= datetime.fromisoformat(r["timestamp"]) < end:
                    out[r["trace_id"]] = float(r["value"])
        return out

    def clear(self, environment: str) -> None:
        d = self.dir / environment
        if d.exists():
            for p in d.glob("*.jsonl"):
                p.unlink()

    def flush(self) -> None:
        pass


class LangfuseBackend:
    """Minimal Langfuse client over the public REST API (no SDK dependency)."""

    name = "langfuse"

    def __init__(self, host: str, public_key: str, secret_key: str, batch_size: int = 50,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.host = host.rstrip("/")
        self._auth = httpx.BasicAuth(public_key, secret_key)
        self._client = httpx.Client(base_url=self.host, auth=self._auth, timeout=30, transport=transport)
        self.batch_size = batch_size
        self._q: queue.Queue[dict | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="langfuse-ingest")
        self._worker.start()
        atexit.register(self.flush)

    @staticmethod
    def to_events(record: TraceRecord) -> list[dict]:
        ts = record.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
        events = [{
            "id": uuid.uuid4().hex,
            "timestamp": ts,
            "type": "trace-create",
            "body": {
                "id": record.id,
                "timestamp": ts,
                "name": record.name,
                "input": record.input,
                "output": record.output,
                "metadata": {**record.metadata, "scores": record.scores},
                "tags": record.tags,
                "environment": record.environment,
            },
        }]
        for name, value in record.scores.items():
            events.append({
                "id": uuid.uuid4().hex,
                "timestamp": ts,
                "type": "score-create",
                "body": {"id": f"{record.id}-{name}", "traceId": record.id, "name": name, "value": float(value),
                         "dataType": "NUMERIC", "environment": record.environment},
            })
        return events

    def write(self, record: TraceRecord) -> None:
        for e in self.to_events(record):
            self._q.put(e)

    def write_feedback(self, trace_id: str, value: float, environment: str, comment: str | None = None) -> None:
        ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self._q.put({"id": uuid.uuid4().hex, "timestamp": ts, "type": "score-create",
                     "body": {"id": f"{trace_id}-user_feedback", "traceId": trace_id, "name": "user_feedback", "value": float(value),
                              "dataType": "NUMERIC", "comment": comment, "environment": environment}})

    def read_feedback(self, start: datetime, end: datetime, environment: str) -> dict[str, float]:
        out, page = {}, 1
        while True:
            r = self._client.get("/api/public/scores", params={
                "name": "user_feedback", "page": page, "limit": 100, "environment": environment,
                "fromTimestamp": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "toTimestamp": end.astimezone(UTC).isoformat().replace("+00:00", "Z")})
            r.raise_for_status()
            data = r.json()
            for s in data.get("data", []):
                out[s["traceId"]] = float(s["value"])
            if page >= int(data.get("meta", {}).get("totalPages", 1) or 1):
                return out
            page += 1

    def _post(self, batch: list[dict]) -> None:
        try:
            r = self._client.post("/api/public/ingestion", json={"batch": batch})
            if r.status_code >= 400:
                logger.warning("Langfuse ingestion failed: %s %s", r.status_code, r.text[:300])
            elif r.status_code == 207 and r.json().get("errors"):
                logger.warning("Langfuse ingestion partial errors: %s", str(r.json()["errors"])[:300])
        except httpx.HTTPError as e:
            logger.warning("Langfuse ingestion error: %s", e)

    def _run(self) -> None:
        batch: list[dict] = []
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                item = "tick"  # type: ignore[assignment]
            if isinstance(item, dict):
                batch.append(item)
            if batch and (item == "tick" or item is None or len(batch) >= self.batch_size):
                self._post(batch)
                batch = []
            if item is None:
                self._q.task_done()
                continue
            if isinstance(item, dict):
                self._q.task_done()

    def flush(self) -> None:
        self._q.put(None)
        self._q.join()

    def read(self, start: datetime, end: datetime, environment: str | None = None) -> list[dict]:
        out: list[dict] = []
        page = 1
        while True:
            params: dict[str, Any] = {
                "page": page, "limit": 100, "name": "rag.ask",
                "fromTimestamp": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "toTimestamp": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            }
            if environment:
                params["environment"] = environment
            r = self._client.get("/api/public/traces", params=params)
            r.raise_for_status()
            data = r.json()
            for t in data.get("data", []):
                out.append(self._normalize(t))
            meta = data.get("meta", {})
            if page >= int(meta.get("totalPages", 1) or 1):
                break
            page += 1
        return sorted(out, key=lambda r: r["timestamp"])

    @staticmethod
    def _normalize(t: dict) -> dict:
        meta = t.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        scores = meta.pop("scores", {}) if isinstance(meta, dict) else {}
        return {
            "id": t.get("id"),
            "timestamp": t.get("timestamp"),
            "environment": t.get("environment", "default"),
            "name": t.get("name"),
            "input": t.get("input") or {},
            "output": t.get("output") or {},
            "metadata": meta,
            "scores": scores,
            "tags": t.get("tags") or [],
        }


class Tracer:
    def __init__(self, backend: TraceBackend, mask: Callable[[str], str] | None = None, environment: str = "production") -> None:
        self.backend, self.mask, self.environment = backend, mask, environment

    def log(self, record: TraceRecord) -> TraceRecord:
        if self.mask is not None:
            record = record.model_copy(update={
                "input": mask_value(record.input, self.mask),
                "output": mask_value(record.output, self.mask),
                "metadata": mask_value(record.metadata, self.mask),
            })
        try:
            self.backend.write(record)
        except Exception as e:  # tracing must never break a request
            logger.warning("trace write failed: %s", e)
        return record

    def read(self, start: datetime, end: datetime, environment: str | None = None) -> list[dict]:
        return self.backend.read(start, end, environment)

    def flush(self) -> None:
        self.backend.flush()


def make_backend(settings: dict | None = None, backend: str | None = None) -> TraceBackend:
    settings = settings or load_settings()
    t = settings.get("tracing", {})
    name = backend or t.get("backend", "auto")
    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY", ""), os.environ.get("LANGFUSE_SECRET_KEY", "")
    if name == "auto":
        name = "langfuse" if pk and sk else "local"
    if name == "langfuse":
        host = os.environ.get("LANGFUSE_HOST") or t.get("langfuse_host", "https://cloud.langfuse.com")
        if not (pk and sk):
            raise RuntimeError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")
        return LangfuseBackend(host, pk, sk)
    if name == "local":
        return LocalBackend(t.get("local_dir", "data/traces"))
    return NullBackend()


def iter_days(start: datetime, end: datetime) -> Iterable[datetime]:
    d = start
    while d < end:
        yield d
        d += timedelta(days=1)
