import re
import streamlit as st
import pandas as pd

EMPTY_MAJOR_DF = pd.DataFrame(columns=["major_code", "major_name", "degree_code", "major_name_raw"])
SYSTEM_MAJOR_LABELS = {
    "Elective Pool",
    "Conditional Pool",
    "Core (Unassigned)",
    "Shared Pool",
}
SYSTEM_MAJOR_LABELS_LOWER = {label.lower() for label in SYSTEM_MAJOR_LABELS}

# Accept friendly labels and normalise to canonical degree codes
DEGREE_ALIASES = {
    "BA": "BA",
    "Bachelor of Arts": "BA",

    "BHlth": "BHlth",
    "Bachelor of Health": "BHlth",
    "Bachelor of Health Sciences": "BHlth",
    "Bachelor of Health Science": "BHlth",

    "BCJ": "BCJ",
    "Bachelor of Criminal Justice": "BCJ",
}

# Map odd codes/typos to canonical codes
CANON_MAP = {
    # Health → BHlth
    "BHSC": "BHlth", "BHSc": "BHlth", "BHS": "BHlth", "BHLTH": "BHlth", "BHLT": "BHlth",
    "BHlth": "BHlth",

    # Arts / CJ
    "BA": "BA",
    "BJC": "BCJ", "BCJ": "BCJ",
}

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


def _update_major_state(
    *,
    majors=None,
    majors_by_degree=None,
    majors_by_degree_raw=None,
    majors_detail=None,
    available=None,
    degree_codes=None,
):
    """Centralise the session-state bookkeeping for majors."""
    st.session_state["majors"] = list(majors or [])
    st.session_state["majors_by_degree"] = {k: list(v) for k, v in (majors_by_degree or {}).items()}
    st.session_state["majors_by_degree_raw"] = {k: list(v) for k, v in (majors_by_degree_raw or {}).items()}
    st.session_state["majors_detail"] = list(majors_detail or [])
    st.session_state["available_majors_by_degree"] = {k: list(v) for k, v in (available or {}).items()}
    st.session_state["degree_codes"] = list(degree_codes or [])


def _apply_major_alias(label: str) -> str:
    name = str(label).strip()
    alias = _MAJOR_ALIAS_MAP.get(name.lower())
    return alias if alias else name


def _strip_major_prefix(label: str) -> str:
    """Remove a leading degree code prefix (e.g. 'BA:') while leaving the rest untouched."""
    if label is None:
        return ""
    text = str(label).strip()
    if not text:
        return text
    match = _MAJOR_PREFIX_RE.match(text)
    if match:
        return _apply_major_alias(match.group(2).strip())
    return _apply_major_alias(text)


# Normalisation utilities.
def _extract_code_or_alias(raw: str) -> str:
    """Resolve a degree label or code to a canonical code (BA / BHlth / BCJ)."""
    s = str(raw or "").strip()
    if not s:
        return ""
    # 1) Code in parentheses
    m = re.search(r"\(([A-Za-z0-9]{2,10})\)", s)
    if m:
        cand = re.sub(r"[^\w]", "", m.group(1))
        return CANON_MAP.get(cand, CANON_MAP.get(cand.upper(), cand))
    # 2) First token as code
    first_tok = re.split(r"[\s(]", s, maxsplit=1)[0]
    first_tok_clean = re.sub(r"[^\w]", "", first_tok)
    if 2 <= len(first_tok_clean) <= 10:
        cand = CANON_MAP.get(first_tok_clean, CANON_MAP.get(first_tok_clean.upper(), first_tok_clean))
        if 2 <= len(cand) <= 10:
            return cand
    # 3) Alias by full name
    name_only = re.sub(r"\(.*?\)", "", s).strip()
    if name_only in DEGREE_ALIASES:
        return DEGREE_ALIASES[name_only]
    compact = re.sub(r"\s+", " ", name_only)
    for k in (name_only, compact, compact.title()):
        if k in DEGREE_ALIASES:
            return DEGREE_ALIASES[k]
    return ""


def _normalise_degree_codes(selected: list[str]) -> list[str]:
    out, seen = [], set()
    for item in selected or []:
        code = _extract_code_or_alias(item)
        if code and code not in seen:
            out.append(code)
            seen.add(code)
    return out


def _canon_degree(d: str) -> str:
    if not d:
        return ""
    d = str(d).strip()
    if d in ("BA", "BHlth", "BCJ"):
        return d
    if d in CANON_MAP:
        return CANON_MAP[d]
    if d in DEGREE_ALIASES:
        return DEGREE_ALIASES[d]
    up = d.upper()
    return CANON_MAP.get(up, DEGREE_ALIASES.get(d, up))


# Data derivation helpers.
def _majors_from_courses_df(df: pd.DataFrame) -> pd.DataFrame:
    """Derive unique majors per degree from a courses dataframe (Degree, Major)."""
    if df is None or df.empty or not {"Degree", "Major"}.issubset(df.columns):
        return EMPTY_MAJOR_DF.copy()
    temp = df[["Degree", "Major"]].dropna().copy()
    temp["degree_code"] = temp["Degree"].map(_canon_degree)
    temp = temp.rename(columns={"Major": "major_name"}).drop_duplicates(["degree_code", "major_name"])
    temp["major_name_raw"] = temp["major_name"]
    temp["major_name"] = temp["major_name"].apply(_strip_major_prefix)
    temp = temp[
        ~temp["major_name"].astype(str).str.strip().str.lower().isin(SYSTEM_MAJOR_LABELS_LOWER)
    ]
    temp = temp[temp["major_name"].astype(str).str.strip() != ""]
    if temp.empty:
        return EMPTY_MAJOR_DF.copy()
    temp = temp.drop_duplicates(["degree_code", "major_name"])
    def mk_code(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", str(name))[:4].upper() or "MAJR"
    temp["major_code"] = temp["major_name"].apply(mk_code)
    return temp[["major_code", "major_name", "degree_code", "major_name_raw"]]


def _load_majors(selected_degrees_codes: list[str]) -> pd.DataFrame:
    """Return majors sourced from the current courses dataframe; no hard-coded fallbacks."""
    if not selected_degrees_codes:
        return EMPTY_MAJOR_DF.copy()

    courses_df = st.session_state.get("courses_df", None)
    derived = _majors_from_courses_df(courses_df)
    if derived.empty:
        return EMPTY_MAJOR_DF.copy()

    cat = derived[derived["degree_code"].isin(selected_degrees_codes)].copy()
    if cat.empty:
        return EMPTY_MAJOR_DF.copy()

    cat["major_name_raw"] = cat["major_name_raw"].fillna(cat["major_name"])
    cat["major_name"] = cat["major_name"].apply(_strip_major_prefix)
    cat = cat.drop_duplicates(["degree_code", "major_name"], keep="first")
    return cat.reset_index(drop=True)


# UI rendering.
def render():
    with st.container(border=True):
        st.subheader("2. Select Your Target Majors")

        selected_from_step1 = st.session_state.get("degrees", [])
        degree_codes = _normalise_degree_codes(selected_from_step1)

        if not degree_codes:
            st.info("No degrees selected.")
            _update_major_state(degree_codes=[])
            return

        st.session_state["degree_codes"] = degree_codes

        majors_df = _load_majors(degree_codes)

        if majors_df.empty and set(degree_codes) == {"BCJ"}:
            st.info("BCJ does not have any majors. Please generate course list.")
            _update_major_state(
                majors_by_degree={"BCJ": []},
                majors_by_degree_raw={"BCJ": []},
                available={"BCJ": []},
                degree_codes=degree_codes,
            )
            return

        if majors_df.empty:
            st.warning("No majors available for the chosen degree(s). (Check degree names vs codes.)")
            _update_major_state(degree_codes=degree_codes)
            return

        # Save AVAILABLE majors by degree for Step 3 (catalogue, not user selection)
        from collections import defaultdict
        available = defaultdict(list)
        for _, row in majors_df.iterrows():
            available[row["degree_code"]].append(row["major_name"])
        st.session_state["available_majors_by_degree"] = dict(available)

        # Build a grouped multiselect so majors sit under their degree.
        majors_df = majors_df.sort_values(["degree_code", "major_name"]).copy()
        majors_df["label"] = majors_df["degree_code"] + " — " + majors_df["major_name"]

        label_to_code = dict(zip(majors_df["label"], majors_df["major_code"]))
        labels = list(label_to_code.keys())

        # Reset selection if the degree set changed
        deg_fp = "|".join(degree_codes)
        if st.session_state.get("_majors_deg_fp") != deg_fp:
            st.session_state.pop("majors_labels", None)
            st.session_state["majors"] = []
            st.session_state["_majors_deg_fp"] = deg_fp

        prev_codes = set(st.session_state.get("majors", []))
        default_labels = (
            [lbl for lbl, code in label_to_code.items() if code in prev_codes]
            if "majors_labels" not in st.session_state
            else []
        )

        chosen_labels = st.multiselect(
            "Select major(s)",
            options=labels,
            default=default_labels,
            key="majors_labels",
            help="Labels are grouped by degree and sorted alphabetically."
        )

        # Map labels -> codes and store clean keys
        chosen_codes = [label_to_code[lbl] for lbl in chosen_labels]
        st.session_state["majors"] = chosen_codes

        # If ONLY BCJ is selected and user picked no majors, still proceed.
        if not chosen_codes and set(degree_codes) == {"BCJ"}:
            _update_major_state(
                majors=[],
                majors_by_degree={"BCJ": []},
                majors_by_degree_raw={"BCJ": []},
                majors_detail=[],
                available=st.session_state.get("available_majors_by_degree", {"BCJ": []}),
                degree_codes=degree_codes,
            )
            return

        selected_df = majors_df[majors_df["major_code"].isin(chosen_codes)].copy()
        detail_cols = ["major_code", "major_name", "degree_code"]
        if "major_name_raw" in selected_df.columns:
            detail_cols.append("major_name_raw")

        by_degree = defaultdict(list)
        by_degree_raw = defaultdict(list)
        for _, row in selected_df.iterrows():
            by_degree[row["degree_code"]].append(row["major_name"])
            raw_label = row["major_name_raw"] if "major_name_raw" in row else row["major_name"]
            by_degree_raw[row["degree_code"]].append(raw_label)

        _update_major_state(
            majors=chosen_codes,
            majors_by_degree=by_degree,
            majors_by_degree_raw=by_degree_raw,
            majors_detail=selected_df[detail_cols].to_dict("records"),
            available=st.session_state.get("available_majors_by_degree", {}),
            degree_codes=degree_codes,
        )

        if chosen_codes:
            st.caption(f"Selected majors: {', '.join(chosen_codes)}")
