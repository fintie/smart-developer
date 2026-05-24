from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Iterator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
SessionLocal = None


def is_database_enabled() -> bool:
    return bool(DATABASE_URL)


if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


@contextmanager
def get_session() -> Iterator[Session]:
    if SessionLocal is None:
        raise RuntimeError(
            "MLOps database logging is disabled because DATABASE_URL is not set."
        )

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()