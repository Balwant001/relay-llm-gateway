from datetime import datetime, timezone
from functools import partial

import anyio.to_thread
from sqlalchemy import (Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint,
                        create_engine, func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from . import config as cfg

_args = {"check_same_thread": False} if cfg.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(cfg.DATABASE_URL, connect_args=_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=60)
    monthly_budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UsageLog(Base):
    __tablename__ = "usage_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    key_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(10), default="api")
    requested_model: Mapped[str] = mapped_column(String(60))
    model: Mapped[str | None] = mapped_column(String(60), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cache: Mapped[str] = mapped_column(String(10), default="none")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    saved_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(10), default="ok")
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    prompt_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    prompt_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Prompt(Base):
    __tablename__ = "prompts"
    __table_args__ = (UniqueConstraint("name", "version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    version: Mapped[int] = mapped_column(Integer)
    template: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    spec: Mapped[str] = mapped_column(Text)
    results: Mapped[str] = mapped_column(Text)


def init_db() -> None:
    Base.metadata.create_all(engine)


async def run(fn, *args, **kwargs):
    """Run a blocking DB function in a worker thread so the event loop stays free."""
    return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))


# ---- small synchronous helpers (always call through run()) ----
def write_log(**kw) -> None:
    with SessionLocal() as s:
        s.add(UsageLog(**kw))
        s.commit()


def write_logs(rows: list[dict]) -> None:
    with SessionLocal() as s:
        s.add_all([UsageLog(**r) for r in rows])
        s.commit()


def lookup_key(key_hash: str):
    with SessionLocal() as s:
        k = s.scalar(select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True)))
        return None if k is None else {"id": k.id, "name": k.name, "rate_limit": k.rate_limit_per_min,
                                       "budget": k.monthly_budget_usd}


def month_spend(key_id: int) -> float:
    start = _now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with SessionLocal() as s:
        return float(s.scalar(select(func.coalesce(func.sum(UsageLog.cost_usd), 0.0)).where(
            UsageLog.key_id == key_id, UsageLog.created_at >= start)) or 0.0)


def get_prompt(name: str, version: int | None):
    with SessionLocal() as s:
        q = select(Prompt).where(Prompt.name == name)
        q = q.where(Prompt.version == version) if version else q.order_by(Prompt.version.desc())
        p = s.scalars(q).first()
        return None if p is None else {"name": p.name, "version": p.version, "template": p.template}
