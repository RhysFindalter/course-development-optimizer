from __future__ import annotations

from contextlib import contextmanager
from typing import Tuple, List
import re

import pandas as pd
import streamlit as st

# --- psycopg v3 replacements ---
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row


# Cache a single pool across Streamlit reruns.
@st.cache_resource(show_spinner=False)
def _get_pool() -> ConnectionPool:
    """
    One global pool per Streamlit process. Re-used across reruns.
    st.secrets["neon"] must contain: host, database, user, password
    """
    cfg = st.secrets["neon"]

    # DSN/conninfo for psycopg v3
    dsn = (
        f"host={cfg['host']} "
        f"dbname={cfg['database']} "
        f"user={cfg['user']} "
        f"password={cfg['password']} "
        f"sslmode=require "
        f"connect_timeout=5 "
        f"application_name=course-optimizer "
        # TCP keepalives (harmless if server ignores)
        f"keepalives=1 keepalives_idle=30 keepalives_interval=10 keepalives_count=5"
    )

    # psycopg_pool handles pooling; autocommit to match your previous behavior
    pool = ConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True},
    )
    return pool


def _ping(conn) -> bool:
    """
    Return True if the connection is healthy.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return True
    except Exception:
        return False


@contextmanager
def get_conn():
    """
    Context-managed connection from the pool.
    Ensures autocommit + tries to set session read-only; recycles broken sockets.
    """
    pool = _get_pool()

    # ConnectionPool provides a context manager for a checked-out connection
    with pool.connection() as conn:
        # Try to set read-only; ignore if role/server disallows
        try:
            with conn.cursor() as cur:
                cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")
        except Exception:
            pass

        # If ping fails, reset pool once and retry
        if not _ping(conn):
            try:
                pool.close()
            except Exception:
                pass
            # Recreate a fresh pool and connection
            pool = _get_pool()
            with pool.connection() as conn2:
                try:
                    with conn2.cursor() as cur:
                        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")
                except Exception:
                    pass
                _ = _ping(conn2)
                yield conn2
                return

        yield conn


def health_check() -> Tuple[bool, str]:
    """
    Lightweight connectivity check + identity. Message includes error type for fast triage.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user;")
                who = cur.fetchone()[0]
        return True, f"OK (as {who})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


