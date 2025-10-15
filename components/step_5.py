import inspect

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

ALIASES = {
    "course code": "Course Code", "code": "Course Code", "course": "Course Code",
    "course name": "Course Name", "name": "Course Name",
    "enrolment score": "Enrolment Score", "enrollment score": "Enrolment Score",
    "enrol score": "Enrolment Score", "enrolment": "Enrolment Score", "enrollment": "Enrolment Score",
    "satisfaction score": "Satisfaction Score", "satisfaction": "Satisfaction Score", "sat score": "Satisfaction Score",
    "overlap score": "Overlap Score", "overlap": "Overlap Score",
    "final score": "Final Score",
    "optimised score": "Optimised Score", "optimized score": "Optimised Score",
    "overall": "Final Score", "score": "Final Score",
    "degree": "Degree", "primary major": "Primary Major", "major": "Major",
}

NUM_COLS = ["Enrolment Score", "Satisfaction Score", "Overlap Score", "Final Score", "Optimised Score"]
COLOR_CANDIDATES = ["Degree", "Primary Major", "Major"]
REQUIRED_BASE = {"Course Code", "Enrolment Score", "Satisfaction Score", "Overlap Score"}

try:
    _PLOTLY_CHART_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
except (ValueError, TypeError):
    _PLOTLY_CHART_SUPPORTS_WIDTH = False

def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())

def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename, seen = {}, set()
    for c in df.columns:
        tgt = ALIASES.get(_norm(c), c)
        if tgt in seen and tgt != c:
            tgt = f"{tgt} (dup)"
        seen.add(tgt)
        rename[c] = tgt
    return df.rename(columns=rename)

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in (set(NUM_COLS) & set(out.columns)):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _looks_like_majors_catalogue(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"major_code", "major_name", "degree_code"}.issubset(cols) and "Course Code" not in cols

def render(data: pd.DataFrame, degrees) -> None:
    with st.container(border=True):
        st.subheader("5. Data visualisation")
        st.write("Explore patterns, clusters, outliers, and trade-offs across enrolment, satisfaction, overlap, and the overall score.")

        plan = st.session_state.get("course_plan")

        df_in = data if isinstance(data, pd.DataFrame) else None
        if df_in is not None and not df_in.empty and _looks_like_majors_catalogue(df_in):
            df_in = None

        if df_in is None or df_in.empty:
            if plan is None or plan.empty:
                st.info("Please generate a course list.")
                return
            df = plan.copy()
        else:
            df = df_in.copy()

        df = _standardise_columns(df)
        df = _coerce_numeric(df)

        final_col = "Final Score" if "Final Score" in df.columns else ("Optimised Score" if "Optimised Score" in df.columns else None)

        missing = sorted(REQUIRED_BASE - set(df.columns))
        if final_col is None:
            missing.append("Final Score (or Optimised Score)")
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.caption(f"Found columns: {list(df.columns)}")
            return

        if final_col != "Final Score":
            df["Final Score"] = df[final_col]

        st.markdown("##### Filters")
        f1, f2, f3, f4 = st.columns([1, 1, 1.2, 1.2])

        enrol_min = 0 if df["Enrolment Score"].isna().all() else int(np.nanmin(df["Enrolment Score"]))
        enrol_max = 100 if df["Enrolment Score"].isna().all() else int(np.nanmax(df["Enrolment Score"]))
        sat_min   = 0.0 if df["Satisfaction Score"].isna().all() else float(np.nanmin(df["Satisfaction Score"]))
        sat_max   = 5.0 if df["Satisfaction Score"].isna().all() else float(np.nanmax(df["Satisfaction Score"]))

        min_enrol = f1.slider("Min enrolment", enrol_min, enrol_max, enrol_min)
        min_sat   = f2.slider("Min satisfaction", sat_min, sat_max, sat_min, step=0.1)

        degree_vals = df["Degree"].dropna().unique().tolist() if "Degree" in df.columns else []
        sel_deg = f3.multiselect("Degree", degree_vals, default=[])

        major_col = "Primary Major" if "Primary Major" in df.columns else ("Major" if "Major" in df.columns else None)
        major_vals = df[major_col].dropna().unique().tolist() if major_col else []
        sel_maj = f4.multiselect("Major", major_vals, default=[])

        g1, g2 = st.columns([1.6, 1])
        q = g1.text_input("Search (code or name contains…)", "")
        colour_options = [c for c in COLOR_CANDIDATES if c in df.columns]
        default_colour = colour_options.index("Degree") if "Degree" in colour_options else 0
        colour_by = g2.selectbox("Colour by", colour_options, index=default_colour if colour_options else 0)

        filtered = df[(df["Enrolment Score"] >= min_enrol) & (df["Satisfaction Score"] >= min_sat)].copy()
        if sel_deg:
            filtered = filtered[filtered["Degree"].isin(sel_deg)]
        if sel_maj and major_col:
            filtered = filtered[filtered[major_col].isin(sel_maj)]
        if q.strip():
            ql = q.strip().lower()
            code_match = filtered["Course Code"].str.lower().str.contains(ql, na=False)
            if "Course Name" in filtered.columns:
                name_match = filtered["Course Name"].str.lower().str.contains(ql, na=False)
                filtered = filtered[code_match | name_match]
            else:
                filtered = filtered[code_match]

        if filtered.empty:
            st.info("No courses match the current filters.")
            return

        st.markdown("###### Scatter controls")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        num_opts = ["Enrolment Score", "Satisfaction Score", "Overlap Score", "Final Score"]
        x_axis = c1.selectbox("X axis", num_opts, index=num_opts.index("Enrolment Score"))
        y_axis = c2.selectbox("Y axis", num_opts, index=num_opts.index("Satisfaction Score"))
        size_by = c3.selectbox("Size by", ["Overlap Score", "Final Score"], index=0)
        show_labels = c4.toggle("Show course labels", value=True)

        x_thr = float(filtered[x_axis].median())
        y_thr = float(filtered[y_axis].median())

        hover_cols = [c for c in ["Course Name", "Degree", "Primary Major", "Major", "Final Score", "Overlap Score"]
                      if c in filtered.columns and c not in (x_axis, y_axis, size_by)]

        fig = px.scatter(
            filtered,
            x=x_axis, y=y_axis, size=size_by, size_max=20,
            text="Course Code" if show_labels else None,
            color=None if colour_by == "(none)" else colour_by,
            hover_data=hover_cols,
            title=f"{x_axis} vs {y_axis} (size = {size_by})",
        )
        fig.update_traces(marker=dict(line=dict(width=0.5, color="#333")))
        if show_labels:
            fig.update_traces(textposition="top center")
        fig.add_vline(x=x_thr, line_dash="dash", line_width=1, opacity=0.5)
        fig.add_hline(y=y_thr, line_dash="dash", line_width=1, opacity=0.5)
        fig.update_layout(
            margin=dict(l=0, r=0, t=48, b=0),
            xaxis_title=x_axis, yaxis_title=y_axis,
            legend_title=(colour_by if colour_by != "(none)" else None),
        )
        plot_kwargs = {"width": "stretch"} if _PLOTLY_CHART_SUPPORTS_WIDTH else {"use_container_width": True}
        st.plotly_chart(fig, **plot_kwargs)
