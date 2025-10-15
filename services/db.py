from __future__ import annotations

from contextlib import contextmanager
from typing import Tuple, List
import re

import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import DictCursor
import streamlit as st

# Cache a single pool across Streamlit reruns.
@st.cache_resource(show_spinner=False)
def _get_pool() -> SimpleConnectionPool:
    """
    One global pool per Streamlit process. Re-used across reruns.
    st.secrets["neon"] must contain: host, database, user, password
    """
    cfg = st.secrets["neon"]
    return SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=cfg["host"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        sslmode="require",
        connect_timeout=5,
        # libpq TCP keepalives to prevent idle drops behind NAT/LB
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        application_name="course-optimizer",
    )

def _ping(conn) -> bool:
    """
    Return True if the connection is healthy; False otherwise.
    Rolls back any failed tx state before checking.
    """
    try:
        try:
            conn.rollback()
        except Exception:
            pass
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return True
    except Exception:
        return False

@contextmanager
def get_conn():
    """
    Context-managed connection from the pool.
    Ensures autocommit + read-only (when supported). Recycles broken sockets.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = True
        try:
            conn.set_session(readonly=True)
        except Exception:
            pass

        if not _ping(conn):
            pool.putconn(conn, close=True)
            conn = pool.getconn()
            conn.autocommit = True
            try:
                conn.set_session(readonly=True)
            except Exception:
                pass
            _ping(conn)

        yield conn
    finally:
        try:
            pool.putconn(conn)
        except Exception:
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass

def health_check() -> Tuple[bool, str]:
    """
    Lightweight connectivity check + identity. Message includes error type for fast triage.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT current_user;")
                who = cur.fetchone()[0]
        return True, f"OK (as {who})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# High-level fetch helpers

_TAG_FALLBACK_LABELS = {
    "E": "Elective Pool",
    "C": "Conditional",
    "R": "Required",
}
_TAG_DEFAULT_LABEL = "Shared Pool"
_TAG_ORDER = ["R", "C", "E"]
_TAG_SET = set(_TAG_ORDER)

_SYSTEM_MAJOR_LABELS = {
    "Elective Pool",
    "Conditional Pool",
    "Core (Unassigned)",
    "Shared Pool",
}
_SYSTEM_MAJOR_LABELS_LOWER = {label.lower() for label in _SYSTEM_MAJOR_LABELS}
_REQ_TYPE_WEIGHT = {
    "R": 100.0,
    "C": 25.0,
    "E": 0.0,
}


def _system_label_series(df: pd.DataFrame, major_col: str) -> pd.Series:
    """
    Human-friendly label (Elective/Conditional/Core) for each course row. Requirement Tag wins
    when the course belongs to a genuine major; otherwise default to Elective Pool.
    """
    if df is None or df.empty:
        return pd.Series(dtype="object")

    labels = pd.Series(_TAG_FALLBACK_LABELS["E"], index=df.index, dtype="object")

    if major_col not in df.columns or "Requirement Tag" not in df.columns:
        return labels

    majors = (
        df[major_col]
        .astype("object")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    real_major_mask = majors.ne("") & ~majors.isin(_SYSTEM_MAJOR_LABELS_LOWER)

    req = (
        df["Requirement Tag"]
        .astype("object")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    req_mask = real_major_mask & req.isin(_TAG_SET)
    if req_mask.any():
        labels.loc[req_mask] = req.loc[req_mask].map(_TAG_FALLBACK_LABELS)

    return labels.fillna(_TAG_FALLBACK_LABELS["E"])


def _compute_requirement_type(df: pd.DataFrame, major_col: str) -> pd.Categorical:
    """
    Ordered categorical (R > C > E). Requirement Tag is used for genuine majors; otherwise default to E.
    """
    if df is None or df.empty:
        return pd.Categorical([], categories=_TAG_ORDER, ordered=True)
    if major_col not in df.columns:
        return pd.Categorical(["E"] * len(df), categories=_TAG_ORDER, ordered=True)

    tags = pd.Series("E", index=df.index, dtype="object")

    if "Requirement Tag" in df.columns:
        majors = (
            df[major_col]
            .astype("object")
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        real_major_mask = majors.ne("") & ~majors.isin(_SYSTEM_MAJOR_LABELS_LOWER)

        req = (
            df["Requirement Tag"]
            .astype("object")
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        req_mask = real_major_mask & req.isin(_TAG_SET)
        if req_mask.any():
            tags.loc[req_mask] = req.loc[req_mask]

    return pd.Categorical(tags, categories=_TAG_ORDER, ordered=True)


def _apply_requirement_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update Requirement Type Score and Optimised Score (base + weight).
    Idempotent: subtracts any previous weight before re-applying.
    """
    if df is None or df.empty or "Requirement Type" not in df.columns or "Optimised Score" not in df.columns:
        return df

    weight = (
        df["Requirement Type"]
        .astype("object")
        .map(_REQ_TYPE_WEIGHT)
        .fillna(0.0)
        .astype(float)
    )
    prev_series = df.get("Requirement Type Score")
    if isinstance(prev_series, (int, float)):
        prev_series = pd.Series(prev_series, index=df.index)
    prev_weight = (
        pd.to_numeric(prev_series, errors="coerce")
        if prev_series is not None else pd.Series(0.0, index=df.index)
    )
    prev_weight = prev_weight.fillna(0.0)
    optim = pd.to_numeric(df["Optimised Score"], errors="coerce").fillna(0.0)
    base = optim - prev_weight

    df["Requirement Type Score"] = weight
    df["Optimised Score"] = base + weight

    if isinstance(df["Requirement Type"], pd.Categorical):
        df["Requirement Type"] = df["Requirement Type"].cat.set_categories(_TAG_ORDER, ordered=True)
    else:
        df["Requirement Type"] = pd.Categorical(df["Requirement Type"], categories=_TAG_ORDER, ordered=True)

    df.sort_values(
        by=["Optimised Score", "Requirement Type Score", "Course Code"],
        ascending=[False, False, True],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_plan_df(bachelor: str, major: str) -> pd.DataFrame:
    """
    Returns a scored plan for a single (bachelor, major).
    Overlap Score = number of DISTINCT degrees (within the selected set) the course has enrolments for,
    computed from course_data.enrolments.
    """
    degree_variants = _expand_degree_variants([bachelor])
    if not degree_variants:
        return pd.DataFrame(columns=[
            "Level", "Course Code", "Primary Major", "Enrolment Score",
            "Satisfaction Score", "Overlap Score", "Optimised Score",
            "Level Rank", "Requirement Tag", "Course Tag"
        ])

    with get_conn() as conn:
        course_source_cte = _course_source_cte()
        sql = f"""
            WITH weights AS (
                SELECT
                    0.01::float8 AS enrol_w,
                    2.0::float8  AS sat_w,
                    1.0::float8  AS ovlp_w
            ),
            overlap_counts AS (
                -- Overlap = number of DISTINCT degrees (from the selected set) with enrolments
                SELECT
                    en.course_code,
                    COUNT(DISTINCT en.bachelor)::int AS overlap_cnt
                FROM course_data.enrolments en
                WHERE en.bachelor = ANY(%(degrees)s::text[])
                GROUP BY en.course_code
            ),
            {course_source_cte}
            base AS (
                SELECT
                    cs.bachelor,
                    cs.major                                          AS primary_major,
                    cs.course_code,
                    course_data.course_level(cs.course_code)          AS level_num,
                    CASE course_data.course_level(cs.course_code)
                        WHEN 100 THEN '100-level'
                        WHEN 200 THEN '200-level'
                        WHEN 300 THEN '300-level'
                        ELSE COALESCE(course_data.course_level(cs.course_code)::text,'Unknown')||'-level'
                    END                                               AS level_label,
                    cs.req_tag,
                    cs.course_tag,
                    es.eval_score                                     AS satisfaction_1_5,   -- 1..5
                    oc.overlap_cnt                                    AS overlap_cnt,        -- DISTINCT degrees with enrolments
                    en.enrolments                                     AS enrolments_raw      -- int
                FROM course_source cs
                LEFT JOIN course_data.evaluation_scores es
                  ON es.course_code = cs.course_code
                LEFT JOIN overlap_counts oc
                  ON oc.course_code = cs.course_code
                LEFT JOIN course_data.enrolments en
                  ON en.course_code = cs.course_code AND en.bachelor = cs.bachelor
                WHERE cs.bachelor = ANY(%(bachelors)s::text[]) AND cs.major = %(major)s
            ),
            scored AS (
                SELECT
                    b.*,
                    ROUND((
                        (SELECT enrol_w FROM weights) * COALESCE(b.enrolments_raw, 0)
                      + (SELECT  sat_w FROM weights) * COALESCE(b.satisfaction_1_5, 0)
                      + (SELECT ovlp_w FROM weights) * COALESCE(b.overlap_cnt, 0)
                    )::numeric, 2) AS optimised_score
                FROM base b
            )
            SELECT
                level_label                             AS "Level",
                course_code                             AS "Course Code",
                primary_major                           AS "Primary Major",
                COALESCE(enrolments_raw, 0)             AS "Enrolment Score",
                ROUND(COALESCE(satisfaction_1_5, 0)::numeric, 2) AS "Satisfaction Score",
                COALESCE(overlap_cnt, 0)                AS "Overlap Score",
                optimised_score                         AS "Optimised Score",
                ROW_NUMBER() OVER (
                    PARTITION BY bachelor, primary_major, level_label
                    ORDER BY optimised_score DESC, course_code
                ) AS "Level Rank",
                req_tag                                  AS "Requirement Tag",
                course_tag                               AS "Course Tag"
            FROM scored
            ORDER BY "Level","Level Rank";
        """
        df = pd.read_sql_query(
            sql,
            conn,
            params={"bachelors": degree_variants, "major": major, "degrees": degree_variants},
        )

    if "Primary Major" in df.columns:
        primary_mask = df["Primary Major"].isna() | df["Primary Major"].astype(str).str.strip().eq("")
        if primary_mask.any():
            fallback = _system_label_series(df, "Primary Major")
            df.loc[primary_mask, "Primary Major"] = fallback.loc[primary_mask]
        df["Primary Major (Raw)"] = df["Primary Major"]
        df["Primary Major"] = df["Primary Major"].apply(_strip_degree_prefix)

    df["Requirement Type"] = _compute_requirement_type(df, "Primary Major")
    df = _apply_requirement_weight(df)

    return df

def fetch_courses_df_for_degrees(degrees: List[str]) -> pd.DataFrame:
    """
    Returns the exact columns downstream steps expect (including DB-calculated "Optimised Score").
    Overlap Score = number of DISTINCT degrees (within the selected set) the course has enrolments for,
    computed from course_data.enrolments.
    Pass DB degree codes, e.g. ["BA", "BHlth", "BCJ"].
    """
    if not degrees:
        return pd.DataFrame(columns=[
            "Course Code", "Course Name", "Degree", "Major",
            "Requirement Tag", "Course Tag",
            "Enrolment Score", "Satisfaction Score", "Overlap Score", "Optimised Score"
        ])

    degrees_db = _expand_degree_variants(degrees)
    if not degrees_db:
        return pd.DataFrame(columns=[
            "Course Code", "Course Name", "Degree", "Major",
            "Requirement Tag", "Course Tag",
            "Enrolment Score", "Satisfaction Score", "Overlap Score", "Optimised Score"
        ])

    with get_conn() as conn:
        course_source_cte = _course_source_cte()
        sql = f"""
            WITH weights AS (
                SELECT
                    0.01::float8 AS enrol_w,
                    2.0::float8  AS sat_w,
                    1.0::float8  AS ovlp_w
            ),
            overlap_counts AS (
                -- Overlap = number of DISTINCT degrees (from the selected set) with enrolments
                SELECT
                    en.course_code,
                    COUNT(DISTINCT en.bachelor)::int AS overlap_cnt
                FROM course_data.enrolments en
                WHERE en.bachelor = ANY(%(degrees)s::text[])
                GROUP BY en.course_code
            ),
            {course_source_cte}
            base AS (
              SELECT
                cs.course_code                         AS "Course Code",
                NULL::text                             AS "Course Name",       -- (placeholder)
                cs.bachelor                            AS "Degree",
                cs.major                               AS "Major",
                cs.req_tag                             AS "Requirement Tag",
                cs.course_tag                          AS "Course Tag",
                COALESCE(en.enrolments, 0)             AS enrolments_raw,      -- RAW enrolments (int)
                COALESCE(es.eval_score, 0)             AS satisfaction_1_5,    -- 1..5 (float)
                COALESCE(oc.overlap_cnt, 0)            AS overlap_cnt          -- DISTINCT degrees with enrolments
              FROM course_source cs
              LEFT JOIN course_data.evaluation_scores es  USING (course_code)
              LEFT JOIN overlap_counts oc                 USING (course_code)
              LEFT JOIN course_data.enrolments en
                ON en.course_code = cs.course_code AND en.bachelor = cs.bachelor
              WHERE cs.bachelor = ANY(%(degrees)s::text[])
            ),
            scored AS (
              SELECT
                "Course Code",
                "Course Name",
                "Degree",
                "Major",
                "Requirement Tag",
                "Course Tag",
                enrolments_raw                        AS "Enrolment Score",
                ROUND(satisfaction_1_5::numeric, 2)   AS "Satisfaction Score",
                overlap_cnt                           AS "Overlap Score",
                ROUND((
                    (SELECT enrol_w FROM weights) * COALESCE(enrolments_raw, 0)
                  + (SELECT  sat_w FROM weights) * COALESCE(satisfaction_1_5, 0)
                  + (SELECT ovlp_w FROM weights) * COALESCE(overlap_cnt, 0)
                )::numeric, 2)                        AS "Optimised Score"
              FROM base
            )
            SELECT * FROM scored
            ORDER BY "Optimised Score" DESC, "Course Code";
        """
        df = pd.read_sql_query(sql, conn, params={"degrees": degrees_db})

    if "Major" in df.columns:
        major_mask = df["Major"].isna() | df["Major"].astype(str).str.strip().eq("")
        if major_mask.any():
            fallback = _system_label_series(df, "Major")
            df.loc[major_mask, "Major"] = fallback.loc[major_mask]
        df["Major (Raw)"] = df["Major"]
        df["Major"] = df["Major"].apply(_strip_degree_prefix)

    df["Requirement Type"] = _compute_requirement_type(df, "Major")
    df = _apply_requirement_weight(df)

    return df


def _course_source_cte() -> str:
    """
    Build the shared course_source CTE combining requirements with the master course list,
    and append additional courses that have enrolments for the selected bachelor but are not
    captured in requirements (e.g., shared electives). Ensures Requirement Tag survives where
    defined, while Course Tag provides the fallback metadata (R/E/C).
    """
    return """
            course_source AS (
                SELECT
                    'requirement'::text AS source_type,
                    r.bachelor,
                    r.major,
                    r.course_code,
                    r.req_tag,
                    f.course_tag
                FROM course_data.requirements r
                JOIN course_data.full_course_list f USING (course_code)
                WHERE r.bachelor = ANY(%(degrees)s::text[])
                UNION ALL
                SELECT
                    'enrolment'::text AS source_type,
                    en.bachelor,
                    NULL::text AS major,
                    en.course_code,
                    NULL::text AS req_tag,
                    f.course_tag
                FROM course_data.enrolments en
                JOIN course_data.full_course_list f USING (course_code)
                LEFT JOIN course_data.requirements r
                  ON r.course_code = en.course_code AND r.bachelor = en.bachelor
                WHERE en.bachelor = ANY(%(degrees)s::text[])
                  AND r.course_code IS NULL
            ),
    """


def _expand_degree_variants(raw_degrees: List[str]) -> List[str]:
    """
    Return the full set of DB-recognised degree tokens for the supplied degrees.
    Ensures we query using every alias (e.g. BCJ + BJC) so electives/core rows are included.
    """
    canonical: set[str] = set()
    for code in raw_degrees or []:
        canon = _canonical_bachelor(code)
        if canon:
            canonical.add(canon)
    if not canonical:
        return []

    variants: set[str] = set()
    for raw, canon in _DEGREE_DB_MAP.items():
        if canon in canonical:
            variants.add(raw)
    return sorted(variants | canonical)


_DEGREE_PREFIX_RE = re.compile(r"^\s*([A-Za-z]{2,10})\s*:\s*(.+)$")
_DEGREE_DB_MAP = {
    "BA": "BA",
    "BACHELOROFARTS": "BA",
    "BHLTH": "BHSc",
    "BHLT": "BHSc",
    "BHS": "BHSc",
    "BH": "BHSc",
    "BHSC": "BHSc",
    "BHlth": "BHSc",
    "BCJ": "BCJ",
    "BJC": "BCJ",
    "BACHELOROFCRIMINALJUSTICE": "BCJ",
}


def _strip_degree_prefix(label: str) -> str:
    """
    Remove leading degree code prefixes such as 'BA: ' without touching the remainder.
    Returns the original string trimmed if no prefix is detected.
    """
    if label is None:
        return ""
    text = str(label).strip()
    if not text:
        return text
    match = _DEGREE_PREFIX_RE.match(text)
    if match:
        return match.group(2).strip()
    return text


def _canonical_bachelor(code: str) -> str:
    if code is None:
        return ""
    raw = str(code).strip()
    if not raw:
        return ""
    match = re.search(r"\(([A-Za-z0-9]{2,10})\)", raw)
    if match:
        raw = match.group(1)
    token = re.sub(r"[^\w]", "", raw)
    upper = token.upper()
    if upper in _DEGREE_DB_MAP:
        return _DEGREE_DB_MAP[upper]
    if token in _DEGREE_DB_MAP:
        return _DEGREE_DB_MAP[token]
    return upper or token
