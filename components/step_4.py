import streamlit as st
import pandas as pd
import re
from typing import Iterable, Dict, List, Optional, Set

SYSTEM_MAJOR_LABELS = {
    "Elective Pool",
    "Conditional Pool",
    "Core (Unassigned)",
    "Shared Pool",
}
SYSTEM_MAJOR_LABELS_LOWER = {label.lower() for label in SYSTEM_MAJOR_LABELS}
_REQ_TYPE_ORDER = ["R", "C", "E"]
_REQ_TYPE_WEIGHT = {
    "R": 100.0,
    "C": 25.0,
    "E": 0.0,
}
_REQ_TYPE_CODES = {  # ensures lexicographic ordering R→C→E in the UI
    "R": "1_R",
    "C": "2_C",
    "E": "3_E",
}
_REQ_TYPE_OPTIONS = list(_REQ_TYPE_CODES.values())
_REQ_TYPE_DISPLAY = {code: code.split("_")[1] for code in _REQ_TYPE_OPTIONS}
_REQ_COL_CONFIG = {
    "Requirement Type": st.column_config.SelectboxColumn(
        "Requirement Type",
        options=_REQ_TYPE_OPTIONS,
        format_func=lambda code: _REQ_TYPE_DISPLAY.get(code, code),
        disabled=True,
    )
}

# PDF export is optional; keep the app usable if the helper is missing.
try:
    from utils.report import build_pdf_report  # type: ignore
except Exception:
    build_pdf_report = None  # Fallback if unavailable

# Column metadata shared across Step 4.
REQUIRED_COLS = {
    "Course Code", "Course Name", "Degree", "Major",
    "Enrolment Score", "Satisfaction Score", "Overlap Score"
}
OPTIONAL_COLS = {"Status", "Requirement Type"}  # Will show if present

TOP_N = 10
MAX_PER_LEVEL = 20

# Numeric columns that should be rounded before display.
NUMERIC_COLS = [
    "Enrolment Score", "Satisfaction Score", "Overlap Score",
    "Requirement Type Score", "Optimised Score"
]

# Mapping helpers for degree labels.
_DEGREE_LABELS = {"BHSc": "BHlth", "BCJ": "BCJ", "BA": "BA"}
_CANON_MAP_IN  = {"BHlth": "BHSc", "BJC": "BCJ", "BHSc": "BHSc", "BCJ": "BCJ", "BA": "BA"}

_MAJOR_PREFIX_RE = re.compile(r"^\s*([A-Za-z]{2,10})\s*:\s*(.+)$")

_MAJOR_ALIAS_MAP = {
    "human services minor": "Media and Communication Major & Human Services Minor",
    "media and communication": "Media and Communication Major & Human Services Minor",
    "media and communication major": "Media and Communication Major & Human Services Minor",
    "media and communication minor": "Media and Communication Major & Human Services Minor",
    "political science major": "Political Science Major & Philosophy Minor",
    "political science": "Political Science Major & Philosophy Minor",
    "philosophy minor": "Political Science Major & Philosophy Minor",
    "philsoipy minor": "Political Science Major & Philosophy Minor",
}


def _apply_major_alias(label: str) -> str:
    name = str(label).strip()
    alias = _MAJOR_ALIAS_MAP.get(name.lower())
    return alias if alias else name


def _strip_major_prefix(label: str) -> str:
    """Remove a leading degree code prefix like 'BA:' while keeping the rest intact."""
    if label is None:
        return ""
    text = str(label).strip()
    if not text:
        return text
    match = _MAJOR_PREFIX_RE.match(text)
    if match:
        return _apply_major_alias(match.group(2).strip())
    return _apply_major_alias(text)


def _clean_major_list(seq: Iterable[str]) -> list[str]:
    """Return unique majors with prefixes stripped, preserving order."""
    seen: Set[str] = set()
    cleaned: list[str] = []
    for item in seq or []:
        candidate = _strip_major_prefix(item)
        if candidate and candidate not in seen:
            cleaned.append(candidate)
            seen.add(candidate)
    return cleaned


def _format_requirement_type_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with sortable Requirement Type codes for interactive tables."""
    if "Requirement Type" not in df.columns:
        return df
    out = df.copy()
    series = out["Requirement Type"]
    mapped = series.map(_REQ_TYPE_CODES)
    out["Requirement Type"] = mapped.fillna(_REQ_TYPE_CODES["E"])
    out = out.drop(columns=["Requirement Type Score"], errors="ignore")
    return out


def _normalise_requirement_type_for_major(df: pd.DataFrame, major_label: str) -> pd.DataFrame:
    """
    Ensure Requirement Type is present and reset to Elective ('E') for courses whose Primary Major
    does not match the target major. Keeps an ordered categorical so sorting respects R→C→E.
    """
    if "Requirement Type" not in df.columns or "Primary Major" not in df.columns:
        return df

    target = _strip_major_prefix(major_label or "")
    values = df["Requirement Type"].astype("object").copy()

    if target:
        majors = df["Primary Major"].astype(str).map(_strip_major_prefix)
        mask = majors.ne(target)
        if mask.any():
            values.loc[mask] = "E"
    df["Requirement Type"] = pd.Categorical(values, categories=_REQ_TYPE_ORDER, ordered=True)

    prev_series = df.get("Requirement Type Score")
    if isinstance(prev_series, (int, float)):
        prev_series = pd.Series(prev_series, index=df.index)
    prev_weight = (
        pd.to_numeric(prev_series, errors="coerce")
        if prev_series is not None else pd.Series(0.0, index=df.index)
    )
    prev_weight = prev_weight.fillna(0.0)
    optim_prev = pd.to_numeric(df.get("Optimised Score"), errors="coerce").fillna(0.0)
    base_score = optim_prev - prev_weight
    new_weight = df["Requirement Type"].astype(str).map(_REQ_TYPE_WEIGHT).fillna(0.0).astype(float)
    df["Requirement Type Score"] = new_weight
    df["Optimised Score"] = base_score + new_weight
    if {"Optimised Score", "Requirement Type"}.issubset(df.columns):
        df.sort_values(
            by=["Optimised Score", "Requirement Type"],
            ascending=[False, False],
            inplace=True,
        )
    else:
        df.sort_values(
            by=[col for col in ["Optimised Score"] if col in df.columns],
            ascending=[False],
            inplace=True,
        )
    df.reset_index(drop=True, inplace=True)
    return df


def _round_cols(df: pd.DataFrame, cols: List[str], ndigits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(ndigits)
    return out


def _degrees_with_majors_bullets(degrees: list[str], plan: pd.DataFrame) -> str:
    majors_by_degree = st.session_state.get("majors_by_degree", {})
    lines = []
    for deg in degrees:
        majors: list[str] = []
        if isinstance(majors_by_degree, dict) and deg in majors_by_degree:
            majors = _clean_major_list(majors_by_degree[deg])
        elif "Primary Major" in plan.columns:
            majors = (
                plan.loc[plan["Degree"] == deg, "Primary Major"]
                .dropna().astype(str).map(_strip_major_prefix).sort_values().unique().tolist()
            )
        label = f"**{deg}**" + (f": {', '.join(majors)}" if majors else "")
        lines.append(f"- {label}")
    return "\n".join(lines)


def _wipe_plan(reason: Optional[str] = None) -> None:
    st.session_state.pop("course_plan", None)
    st.session_state.pop("course_plan_degrees", None)
    if reason:
        st.session_state["course_plan_wipe_reason"] = reason


def _ensure_degree(df: pd.DataFrame) -> pd.DataFrame:
    if "Degree" in df.columns:
        return df

    def map_degree(code: str) -> str:
        code = str(code)
        if code.startswith("HLTH"):
            return "BHSc"
        if code.startswith("CRJU"):
            return "BCJ"
        return "BA"

    out = df.copy()
    out["Degree"] = out["Course Code"].astype(str).map(map_degree)
    return out


def _level_band_from_code(code: str) -> str:
    s = str(code)
    m = re.match(r'^[A-Za-z]+(\d)', s)
    if not m:
        return "Other"
    first = m.group(1)
    if first == "1":
        return "100-level"
    if first == "2":
        return "200-level"
    if first == "3":
        return "300-level"
    return "Other"

# Build a per-degree archive with 100/200/300 level slices.
def _build_degree_major_csv_zip(plan: pd.DataFrame,
                                degree_list: List[str],
                                majors_map_raw,
                                min_per_level: int = MAX_PER_LEVEL,
                                include_course_name_in_levels: bool = False) -> bytes:
    """
    Create a ZIP archive containing:
      - One CSV per Degree–Major (or Degree-only if no majors),
        laid out with 100/200/300 via a 'Level' column (Course Name excluded by default).
      Each level is padded (prioritising same-degree courses) and capped at 20 rows,
      matching the PDF course list specification.
    """
    import io, re, zipfile

    def _safe_filename(name: str) -> str:
        s = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', '', str(name))
        s = re.sub(r"\s+", " ", s).strip()
        return s[:60] if len(s) > 60 else s

    plan = plan.copy()
    plan["Degree"] = plan["Degree"].astype(str).str.strip().replace({"BJC": "BCJ"})

    # Canonical degree list (match your PDF/export logic)
    given = [_CANON_MAP_IN.get(d, d) for d in (degree_list or [])]
    from_plan = plan["Degree"].dropna().astype(str).unique().tolist()
    degree_list2: List[str] = []
    for d in (given + sorted(from_plan)):
        if d and d not in degree_list2:
            degree_list2.append(d)

    majors_map = _normalise_majors_map(majors_map_raw)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Emit one CSV per degree/major pair.
        level_order = {"100-level": 0, "200-level": 1, "300-level": 2, "Other": 3}

        def _with_level(df: pd.DataFrame) -> pd.DataFrame:
            tmp = df.copy()
            tmp["Level"] = tmp["Course Code"].astype(str).map(_level_band_from_code)
            return tmp

        for deg in degree_list2:
            display_deg = _DEGREE_LABELS.get(deg, deg)
            sub_deg = plan[plan["Degree"] == deg].copy()

            if deg in majors_map and majors_map[deg]:
                majors = majors_map[deg]
            else:
                majors = (
                    sub_deg["Primary Major"].dropna().astype(str).sort_values().unique().tolist()
                    if "Primary Major" in sub_deg.columns else []
                )

            majors = [
                m for m in majors
                if isinstance(m, str) and m.strip() and m.lower() not in SYSTEM_MAJOR_LABELS_LOWER
            ]

            targets = majors if majors else ["—"]

            for mj in targets:
                base = sub_deg if mj == "—" else sub_deg[sub_deg.get("Primary Major", "") == mj].copy()
                base = _normalise_requirement_type_for_major(base, "" if mj == "—" else mj)

                padded = _ensure_min_courses_per_level(
                    base, plan, deg, min_per_level=min_per_level, max_per_level=MAX_PER_LEVEL
                ).copy()
                padded = _normalise_requirement_type_for_major(padded, "" if mj == "—" else mj)
                padded = _with_level(padded)

                # Add a rank per level for consistent ordering.
                if "Optimised Score" in padded.columns:
                    padded["Level Rank"] = padded.groupby("Level")["Optimised Score"] \
                                                .rank(ascending=False, method="first").astype(int)
                else:
                    padded["Level Rank"] = pd.NA

                # Trim columns; course name is optional for these exports.
                cols_pref = [
                    "Level", "Course Code", "Primary Major", "Requirement Type",
                    "Enrolment Score", "Satisfaction Score", "Overlap Score", "Optimised Score",
                    "Level Rank",
                ]
                if include_course_name_in_levels:
                    cols_pref.insert(2, "Course Name")  # after Course Code

                keep = [c for c in cols_pref if c in padded.columns]
                out = padded[keep].copy()
                if "Requirement Type" in out.columns:
                    out["Requirement Type"] = out["Requirement Type"].astype("string").replace("nan", "").fillna("")

                # Sort levels in order and then apply the within-level rank.
                out["_lvl"] = out["Level"].map(level_order).fillna(99).astype(int)
                out = out.sort_values(["_lvl", "Level Rank"], kind="stable")
                out = out.drop(columns=["_lvl"])

                # Round numeric columns for readability.
                for c in ["Enrolment Score", "Satisfaction Score", "Overlap Score", "Optimised Score"]:
                    if c in out.columns:
                        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
                if "Level Rank" in out.columns:
                    out["Level Rank"] = pd.to_numeric(out["Level Rank"], errors="coerce").astype("Int64")

                # Persist the sheet into the archive.
                base_name = display_deg if mj == "—" else f"{display_deg} — {mj}"
                fname = _safe_filename(base_name) + ".csv"
                zf.writestr(fname, out.to_csv(index=False).encode("utf-8-sig"))

    zip_buf.seek(0)
    return zip_buf.read()


def _show_level_grouped_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No courses to display for this selection.")
        return

    temp = df.copy()
    temp["Level Band"] = temp["Course Code"].astype(str).map(_level_band_from_code)

    # Hide Level Band, Degree, and Course Name from the per-level tables
    hide_cols = {"Level Band", "Degree", "Course Name"}
    show_cols = [c for c in temp.columns if c not in hide_cols]

    # (Optional) keep a sensible order if those columns exist
    preferred = [
        "Course Code", "Primary Major", "Requirement Type",
        "Enrolment Score", "Satisfaction Score", "Overlap Score", "Optimised Score", "Status"
    ]

    ordered_cols = [c for c in preferred if c in show_cols] + [c for c in show_cols if c not in preferred]

    for label in ("100-level", "200-level", "300-level"):
        group = temp[temp["Level Band"] == label]
        if not group.empty:
            # Keep highest scoring courses near the top.
            if "Optimised Score" in group.columns:
                group = group.sort_values("Optimised Score", ascending=False)
            group = group.head(20)
            st.markdown(f"#### {label}")
            display = _round_cols(group[ordered_cols], NUMERIC_COLS)
            display = _format_requirement_type_for_ui(display)
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config=_REQ_COL_CONFIG,
        )

def _ensure_min_courses_per_level(filtered: pd.DataFrame,
                                  plan: pd.DataFrame,
                                  degree_code: str,
                                  min_per_level: int = MAX_PER_LEVEL,
                                  max_per_level: Optional[int] = None) -> pd.DataFrame:
    """
    Pad filtered plan so each level (100/200/300) has at least `min_per_level` courses.
    Optionally cap each level at `max_per_level` (after sorting by Optimised Score).
    """
    if filtered.empty:
        return filtered

    def with_level(df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp["Level Band"] = tmp["Course Code"].astype(str).map(_level_band_from_code)
        return tmp

    base = with_level(filtered)
    full = with_level(plan)
    have_codes = set(base["Course Code"].astype(str))
    out_parts = []

    for level in ("100-level", "200-level", "300-level"):
        group = base[base["Level Band"] == level]
        need = max(0, min_per_level - len(group))
        if need <= 0:
            group_sorted = group.sort_values("Optimised Score", ascending=False) if "Optimised Score" in group.columns else group
            if max_per_level is not None:
                group_sorted = group_sorted.head(max_per_level)
            out_parts.append(group_sorted)
            continue

        picked = pd.DataFrame()

        same_deg_candidates = full[
            (full["Level Band"] == level) & (full["Degree"] == degree_code)
        ]
        same_deg_candidates = same_deg_candidates[
            ~same_deg_candidates["Course Code"].astype(str).isin(have_codes)
        ].sort_values("Optimised Score", ascending=False)

        picked = same_deg_candidates.head(need)
        have_codes.update(picked["Course Code"].astype(str))
        need -= len(picked)

        if need > 0:
            any_deg_candidates = full[(full["Level Band"] == level)]
            any_deg_candidates = any_deg_candidates[
                ~any_deg_candidates["Course Code"].astype(str).isin(have_codes)
            ].sort_values("Optimised Score", ascending=False)
            picked = pd.concat([picked, any_deg_candidates.head(need)], ignore_index=True)

        group_padded = pd.concat([group, picked], ignore_index=True)

        if "Optimised Score" in group_padded.columns:
            group_padded = group_padded.sort_values("Optimised Score", ascending=False, ignore_index=True)
        if max_per_level is not None:
            group_padded = group_padded.head(max_per_level)

        out_parts.append(group_padded)

    combined = pd.concat(out_parts, ignore_index=True)
    if "Level Band" in combined.columns:
        combined = combined.drop(columns=["Level Band"])
    combined = combined[[c for c in filtered.columns if c in combined.columns] +
                        [c for c in combined.columns if c not in filtered.columns]]
    return combined


def _normalise_majors_map(raw_map) -> Dict[str, List[str]]:
    """Normalise majors_by_degree keys to canonical codes (BHlth→BHSc, BJC→BCJ)."""
    norm: Dict[str, List[str]] = {}
    if not isinstance(raw_map, dict):
        return norm
    for k, majors in raw_map.items():
        canon = _CANON_MAP_IN.get(str(k).strip(), str(k).strip())
        if isinstance(majors, (list, tuple)):
            norm[canon] = _clean_major_list(str(m).strip() for m in majors if str(m).strip())
    return norm


def _optimise_courses(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure "Major" column is standardised
    if "Major" in out.columns:
        out = out.rename(columns={"Major": "Primary Major"})
    if "Primary Major" in out.columns:
        out["Primary Major"] = out["Primary Major"].apply(_strip_major_prefix)

    # Make sure we keep the DB-calculated Optimised Score
    if "Optimised Score" not in out.columns:
        raise ValueError("Expected 'Optimised Score' from DB but it was missing.")

    out = out.sort_values("Optimised Score", ascending=False).reset_index(drop=True)

    base_cols = [
        "Course Code", "Course Name", "Degree", "Primary Major", "Requirement Type",
        "Enrolment Score", "Satisfaction Score", "Overlap Score", "Optimised Score",
    ]

    if "Status" in out.columns:
        base_cols.insert(2, "Status")

    keep = [c for c in base_cols if c in out.columns]
    result = out[keep]
    if "Course Name" in result.columns:
        result = result.drop(columns=["Course Name"])
    return result


def render(courses_df: pd.DataFrame, clicked: bool, degrees: list[str]) -> None:
    with st.container(border=True):
        st.subheader("4. Course List")
        st.write("Review a detailed report of the course list and related metrics.")

        if courses_df is None or courses_df.empty:
            st.info("Please generate a course list.")
            return

        courses_df = _ensure_degree(courses_df)

        missing = REQUIRED_COLS - set(courses_df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            return

        if st.session_state.get("clear_all_clicked"):
            _wipe_plan("Cleared via Clear All")
            st.session_state["clear_all_clicked"] = False

        prev_degrees = st.session_state.get("course_plan_degrees")
        if prev_degrees is not None and set(prev_degrees) != set(degrees):
            _wipe_plan("Degree selection changed")

        if clicked:
            st.session_state["course_plan"] = _optimise_courses(courses_df)
            st.session_state["course_plan_degrees"] = list(degrees)

        plan = st.session_state.get("course_plan")
        plan_degrees = st.session_state.get("course_plan_degrees", degrees)
        if not degrees and plan is not None:
            _wipe_plan("No degrees selected")

        plan = st.session_state.get("course_plan")
        if plan is None:
            st.info("Please generate a course list.")
            return

        plan = plan.copy()
        plan = plan.drop_duplicates(subset=["Course Code", "Degree", "Primary Major"], keep="first")
        if "Primary Major (Raw)" in plan.columns:
            plan = plan.drop(columns=["Primary Major (Raw)"])
        if "Primary Major" in plan.columns:
            plan["Primary Major"] = plan["Primary Major"].apply(_strip_major_prefix)
        plan["Degree"] = (
            plan["Degree"].astype(str).str.strip().replace({"BJC": "BCJ"})
        )
        plan = plan[plan["Degree"] != ""]

        majors_selected = 0
        majors_state = st.session_state.get("majors_by_degree", {})
        if isinstance(majors_state, dict):
            for vals in majors_state.values():
                if isinstance(vals, (list, tuple)):
                    majors_selected += len(_clean_major_list(vals))

        c1, c2, c3 = st.columns(3)
        c1.metric("Degrees selected", len(plan_degrees))
        c2.metric("Majors selected", majors_selected)
        c3.metric("Courses in list", len(plan))

        st.markdown("### Summary")
        if plan_degrees:
            st.markdown("**Target degrees with corresponding Majors**")
            st.markdown(_degrees_with_majors_bullets(plan_degrees, plan))
        else:
            st.info("No degrees selected. Add degrees earlier to tailor this report.")

        st.markdown("**Score definitions**")
        st.markdown(
            "- **Optimised Score** = (0.01 * Enrolment Score) + (2.0 * Satisfaction Score) + (1.0 * Overlap Score)\n"
            "- **Enrolment Score** the number of students enrolled in each course from 2021 - 2025\n"
            "- **Satisfaction Score** how satisfied students who took the course were /5\n"
            "- **Overlap Score** how many degrees a course can be used in"
        )

        st.markdown("### Course List by Degree & Major")
        degree_codes = plan["Degree"].dropna().astype(str).str.strip()
        degree_options: list[str] = []

        for code in plan_degrees or []:
            cleaned = str(code).strip()
            mapped = _CANON_MAP_IN.get(cleaned, cleaned)
            if mapped and mapped not in degree_options:
                degree_options.append(mapped)

        for code in sorted(degree_codes.unique()):
            cleaned = _CANON_MAP_IN.get(code, code)
            if cleaned and cleaned not in degree_options:
                degree_options.append(cleaned)

        if not degree_options:
            st.info("No degrees available in the current plan.")
            return

        DEGREE_LABELS = {"BHSc": "BHlth", "BCJ": "BCJ", "BA": "BA"}
        col_deg, col_maj = st.columns([1, 1])
        with col_deg:
            degree_choice = st.selectbox(
                "Choose a Degree",
                degree_options,
                index=0,
                format_func=lambda code: DEGREE_LABELS.get(code, code),
            )

        majors_by_degree = st.session_state.get("majors_by_degree", {})
        majors_by_degree_raw = st.session_state.get("majors_by_degree_raw", {})
        if isinstance(majors_by_degree, dict) and degree_choice in majors_by_degree:
            major_options = [
                m for m in _clean_major_list(majors_by_degree[degree_choice])
                if m and m.strip() and m.lower() not in SYSTEM_MAJOR_LABELS_LOWER
            ]
        elif "Primary Major" in plan.columns:
            major_options = (
                plan.loc[plan["Degree"] == degree_choice, "Primary Major"]
                .dropna().astype(str).map(_strip_major_prefix).sort_values().unique().tolist()
            )
            major_options = [m for m in major_options if m and m.strip()]
            major_options = [m for m in major_options if m.lower() not in SYSTEM_MAJOR_LABELS_LOWER]
        else:
            major_options = []

        with col_maj:
            major_choice = st.selectbox(
                "Choose a Major",
                major_options if major_options else ["(No majors available for this degree)"],
                index=0,
                disabled=(len(major_options) == 0),
            )
        major_choice_clean = _strip_major_prefix(major_choice) if major_options else ""

        filtered = plan[plan["Degree"] == degree_choice].copy()
        if "Primary Major" in filtered.columns:
            filtered["Primary Major"] = filtered["Primary Major"].apply(_strip_major_prefix)

        if "Primary Major" in filtered.columns and major_options:
            if major_choice in major_options:
                filtered = filtered[filtered["Primary Major"] == major_choice_clean]
            else:
                # Combine any majors that map to the same alias
                raw_list = majors_by_degree_raw.get(degree_choice, []) if isinstance(majors_by_degree_raw, dict) else []
                matching_aliases = {
                    _strip_major_prefix(raw)
                    for raw in raw_list
                    if _strip_major_prefix(raw) == major_choice_clean
                }
                if matching_aliases:
                    filtered = filtered[filtered["Primary Major"].isin(matching_aliases)]

        filtered = _normalise_requirement_type_for_major(filtered, major_choice_clean)

        filtered = _ensure_min_courses_per_level(
            filtered, plan, degree_choice, min_per_level=MAX_PER_LEVEL, max_per_level=MAX_PER_LEVEL
        )
        filtered = _normalise_requirement_type_for_major(filtered, major_choice_clean)

        st.markdown(
            f"### Course List for {DEGREE_LABELS.get(degree_choice, degree_choice)}"
            + (f" — {major_choice_clean}" if major_options and major_choice in major_options else "")
        )
        _show_level_grouped_table(filtered)

        st.subheader("Assumptions")
        st.markdown(
            "- Optimisation uses enrolments, student satisfaction, and overlap between degrees.\n"
            "- Recommendations are advisory; final decisions remain with UC Online.\n"
            "- Input course data is current and pre-cleaned."
        )

        st.markdown("### Export")
        if build_pdf_report is not None:
            try:
                pdf_bytes = build_pdf_report(plan, degrees=plan_degrees)
                st.download_button(
                    "Download Report (PDF)",
                    pdf_bytes,
                    file_name="uc_course_build_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            except Exception:
                st.caption("PDF export unavailable (error in utils.report.build_pdf_report).")

        try:
            majors_raw = st.session_state.get("majors_by_degree", {})
            per_dm_zip = _build_degree_major_csv_zip(
                plan,
                degree_list=plan_degrees,
                majors_map_raw=majors_raw,
                min_per_level=MAX_PER_LEVEL,
                include_course_name_in_levels=True,
            )
            st.download_button(
                "Download Course Plan (CSV)",
                per_dm_zip, 
                file_name="uc_course_build_report_tables.zip",
                mime="application/zip",
                width="stretch",
            )
        except Exception as e:
            st.caption(f"Degree–Major CSV export unavailable ({e}).")

