"""
Database connection manager for ProcureLens.

Provides engine creation (singleton), session context manager,
and schema initialisation utilities.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "sqlite:///data/procurelens.db"
)

_engine: Optional[Engine] = None


def get_engine(url: Optional[str] = None) -> Engine:
    """Return a singleton SQLAlchemy engine.

    Parameters
    ----------
    url : str, optional
        Override the default DATABASE_URL for this engine.  Only the first
        call's *url* is used; subsequent calls return the cached engine.

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    global _engine
    if _engine is None:
        effective_url = url or DATABASE_URL
        connect_args = {}
        if effective_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            effective_url,
            echo=False,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
    return _engine


@contextmanager
def get_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Yield a transactional SQLAlchemy session.

    Usage::

        with get_session() as session:
            session.add(obj)

    The session is committed on clean exit, rolled back on exception,
    and always closed.
    """
    eng = engine or get_engine()
    factory = sessionmaker(bind=eng)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None) -> None:
    """Create all tables defined on :pydata:`Base`.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine, optional
        Engine to bind.  Defaults to the singleton from :func:`get_engine`.
    """
    eng = engine or get_engine()
    Base.metadata.create_all(eng)
