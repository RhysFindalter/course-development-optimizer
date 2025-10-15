"""Chart helpers for common course views."""
from __future__ import annotations
import pandas as pd
import plotly.express as px

# These helpers expect core columns like Course Code, Course Name, enrolment, satisfaction, overlap, and final scores.

_NUMERIC_SAFE_DEFAULTS = {
    "Enrolment Score": 0.0,
    "Satisfaction Score": 0.0,
    "Overlap Score": 0.0,
    "Final Score": 0.0,
}

def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in _NUMERIC_SAFE_DEFAULTS.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    return out

def _ensure_degrees_list(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Degrees" in out.columns:
        if out["Degrees"].dtype == object:
            out["Degrees"] = out["Degrees"].apply(
                lambda v: v if isinstance(v, (list, tuple, set)) else
                [s.strip() for s in str(v).split(";")] if pd.notna(v) else []
            )
    else:
        out["Degrees"] = [[]] * len(out)
    return out

def fig_bar_top_courses(df: pd.DataFrame, by: str = "Enrolment Score", top_n: int = 15):
    df = _ensure_numeric(df)
    by = by if by in df.columns else "Enrolment Score"
    df = df.sort_values(by=by, ascending=False).head(top_n)
    fig = px.bar(
        df,
        x=by,
        y="Course Name",
        orientation="h",
        text=by,
        hover_data=[c for c in ["Course Code","Satisfaction Score","Overlap Score","Final Score"] if c in df.columns],
        title=f"Top {min(top_n, len(df))} Courses by {by}",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig
