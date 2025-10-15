"""PDF report generation helpers."""
from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Optional
import os
import unicodedata

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Styling constants.
PAGE_SIZE = A4
MARGINS   = (1.6*cm, 1.2*cm, 1.6*cm, 1.6*cm)
ACCENT    = colors.HexColor("#003A70")
STRIPE    = colors.HexColor("#E9F1FB")
MAX_PER_LEVEL = 20

# Canonical code -> display label
DEGREE_LABELS: Dict[str, str] = {"BHSc": "BHlth", "BCJ": "BCJ", "BA": "BA"}
# Display label or legacy code -> canonical code
CANON_MAP_IN: Dict[str, str] = {"BHlth": "BHSc", "BJC": "BCJ", "BHSc": "BHSc", "BCJ": "BCJ", "BA": "BA"}

# Try to register a Unicode-friendly font so non-ASCII characters survive.
UNICODE_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "C:\\Windows\\Fonts\\arialuni.ttf",
    "C:\\Windows\\Fonts\\DejaVuSans.ttf",
]
FONT_REG  = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
HAS_UNICODE_FONT = False

for path in UNICODE_FONT_CANDIDATES:
    if os.path.exists(path):
        try:
            pdfmetrics.registerFont(TTFont("UnicodeBase", path))
            FONT_REG = "UnicodeBase"
            FONT_BOLD = "UnicodeBase"
            HAS_UNICODE_FONT = True
            break
        except Exception:
            continue


# Helper routines.
def _strip_accents_if_needed(s: str) -> str:
    if HAS_UNICODE_FONT:
        return s
    norm = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _level_band_from_code(code: str) -> str:
    import re
    s = str(code)
    m = re.match(r"^[A-Za-z]+(\d)", s)
    if not m:
        return "Other"
    d = m.group(1)
    return {"1": "100-level", "2": "200-level", "3": "300-level"}.get(d, "Other")


def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = doc.pagesize
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(0.6)
    canvas.line(MARGINS[0], h - MARGINS[2] + 0.6*cm, w - MARGINS[1], h - MARGINS[2] + 0.6*cm)
    canvas.setFont(FONT_BOLD, 9)
    canvas.setFillColor(ACCENT)
    canvas.drawString(MARGINS[0], h - MARGINS[2] + 0.8*cm, "UC Online — Course List Report")
    canvas.setFont(FONT_REG, 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(w - MARGINS[1], MARGINS[3] - 0.5*cm, f"Page {doc.page}")
    canvas.restoreState()


def _styles():
    base = getSampleStyleSheet()
    base["Normal"].fontName = FONT_REG
    base["Normal"].fontSize = 9
    base["Normal"].leading = 12

    title = ParagraphStyle(
        "TitleBig", parent=base["Heading1"], fontName=FONT_BOLD,
        fontSize=20, leading=24, textColor=ACCENT, alignment=TA_LEFT, spaceAfter=6
    )
    h2 = ParagraphStyle(
        "H2", parent=base["Heading2"], fontName=FONT_BOLD,
        fontSize=14, leading=18, textColor=ACCENT, spaceBefore=10, spaceAfter=6,
        keepWithNext=True
    )
    h3 = ParagraphStyle(
        "H3", parent=base["Heading3"], fontName=FONT_BOLD,
        fontSize=12, leading=16, textColor=colors.black, spaceBefore=6, spaceAfter=4
    )
    meta = ParagraphStyle(
        "Meta", parent=base["Normal"], fontSize=9, textColor=colors.grey, leading=12
    )
    cell = ParagraphStyle(
        "Cell", parent=base["Normal"], fontName=FONT_REG, fontSize=8,
        leading=10, spaceAfter=0, allowOrphans=0, allowWidows=0
    )
    cell_bold = ParagraphStyle("CellHeader", parent=cell, fontName=FONT_BOLD)
    return base, title, h2, h3, meta, cell, cell_bold


def _format_df_for_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if "Score" in col or "Enrol" in col or "Satisf" in col or "Overlap" in col or "Optim" in col:
                out[col] = out[col].round(2)
    for col in out.columns:
        out[col] = out[col].astype(str).map(_strip_accents_if_needed)
    return out


def _table(df: pd.DataFrame) -> Table:
    _, _, _, _, _, cell, cell_bold = _styles()
    df = _format_df_for_table(df)
    headers = [Paragraph(_strip_accents_if_needed(h), cell_bold) for h in df.columns]
    rows = [[Paragraph(str(v), cell) for v in row] for row in df.itertuples(index=False)]
    base_widths = [0.9*inch, 1.1*inch, 0.9*inch, 0.95*inch, 0.95*inch, 0.95*inch, 0.9*inch]
    col_count = len(headers)
    col_widths = (base_widths + [0.9*inch] * col_count)[:col_count]
    t = Table([headers] + rows, repeatRows=1, colWidths=col_widths)
    ts = [
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.Color(0, 0, 0, 0.12)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), STRIPE))
    t.setStyle(TableStyle(ts))
    return t


def _with_level(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Level Band"] = tmp["Course Code"].astype(str).map(_level_band_from_code)
    return tmp


def _pad_min_per_level(filtered: pd.DataFrame,
                       plan: pd.DataFrame,
                       degree_code: str,
                       min_per_level: int = MAX_PER_LEVEL,
                       max_per_level: Optional[int] = MAX_PER_LEVEL) -> pd.DataFrame:
    if filtered.empty:
        return filtered
    base = _with_level(filtered)
    full = _with_level(plan)
    have = set(base["Course Code"].astype(str))
    parts: List[pd.DataFrame] = []
    for band in ("100-level", "200-level", "300-level"):
        group = base[base["Level Band"] == band]
        need = max(0, min_per_level - len(group))
        if need <= 0:
            if "Optimised Score" in group.columns:
                group = group.sort_values("Optimised Score", ascending=False)
            if max_per_level is not None:
                group = group.head(max_per_level)
            parts.append(group)
            continue
        picked = pd.DataFrame()
        c_same = full[(full["Level Band"] == band) & (full["Degree"] == degree_code)]
        c_same = c_same[~c_same["Course Code"].astype(str).isin(have)] \
                    .sort_values("Optimised Score", ascending=False)
        pick1 = c_same.head(need)
        have.update(pick1["Course Code"].astype(str))
        need -= len(pick1)
        picked = pick1
        if need > 0:
            c_any = full[(full["Level Band"] == band)]
            c_any = c_any[~c_any["Course Code"].astype(str).isin(have)] \
                      .sort_values("Optimised Score", ascending=False)
            pick2 = c_any.head(need)
            picked = pd.concat([picked, pick2], ignore_index=True)
        group_padded = pd.concat([group, picked], ignore_index=True)
        if "Optimised Score" in group_padded.columns:
            group_padded = group_padded.sort_values("Optimised Score", ascending=False, ignore_index=True)
        if max_per_level is not None:
            group_padded = group_padded.head(max_per_level)
        parts.append(group_padded)
    out = pd.concat(parts, ignore_index=True)
    return out.drop(columns=["Level Band"], errors="ignore")


def _level_table_blocks(df: pd.DataFrame) -> List[List[Flowable]]:
    blocks: List[List[Flowable]] = []
    temp = _with_level(df)
    show_cols = [c for c in temp.columns if c not in ("Level Band", "Degree", "Course Name")]
    _, _, h2, *_ = _styles()
    for band in ("100-level", "200-level", "300-level"):
        sub = temp[temp["Level Band"] == band]
        if sub.empty:
            continue
        sub = sub.drop(columns=["Course Name"], errors="ignore")
        block = [
            Paragraph(_strip_accents_if_needed(band), h2),
            _table(sub[show_cols]),
            Spacer(1, 0.05*inch),
        ]
        blocks.append(block)
    return blocks


def _normalise_majors_map(raw_map) -> Dict[str, List[str]]:
    """Normalise majors_by_degree keys to canonical degree codes (BHlth→BHSc, BJC→BCJ),
    de-dup majors and keep order. Returns {} if input not a dict."""
    norm: Dict[str, List[str]] = {}
    if not isinstance(raw_map, dict):
        return norm
    for k, majors in raw_map.items():
        canon = CANON_MAP_IN.get(str(k).strip(), str(k).strip())
        if isinstance(majors, (list, tuple)):
            # de-dup while preserving order
            norm[canon] = list(dict.fromkeys([str(m).strip() for m in majors if str(m).strip()]))
    return norm


def _degrees_with_majors_bullets(degree_list: List[str], plan: pd.DataFrame, majors_map) -> Paragraph:
    base, *_ = _styles()
    majors_map = _normalise_majors_map(majors_map)
    lines = []
    for deg in degree_list:
        majors = []
        if deg in majors_map and majors_map[deg]:
            majors = majors_map[deg]
        elif "Primary Major" in plan.columns:
            majors = (
                plan.loc[plan["Degree"] == deg, "Primary Major"]
                .dropna().astype(str).sort_values().unique().tolist()
            )
        label = DEGREE_LABELS.get(deg, deg)
        majors_txt = ", ".join(majors) if majors else "—"
        lines.append(f"• <b>{_strip_accents_if_needed(label)}</b>: {_strip_accents_if_needed(majors_txt)}")
    html = "<br/>".join(lines) if lines else "—"
    return Paragraph(html, base["Normal"])


# Public API.
def build_pdf_report(plan: pd.DataFrame,
                     degrees: List[str],
                     *,
                     title_text: str = "Course List Report",
                     min_per_level: int = MAX_PER_LEVEL,
                     top_n_overall: int = 25) -> bytes:
    plan = plan.copy()
    plan["Degree"] = plan["Degree"].astype(str).str.strip().replace({"BJC": "BCJ"})

    given = [CANON_MAP_IN.get(d, d) for d in (degrees or [])]
    from_plan = plan["Degree"].dropna().astype(str).unique().tolist()
    degree_list = []
    for d in (given + sorted(from_plan)):
        if d and d not in degree_list:
            degree_list.append(d)

    try:
        import streamlit as st
        majors_raw = st.session_state.get("majors_by_degree", {})
    except Exception:
        majors_raw = {}
    majors_map = _normalise_majors_map(majors_raw)  # <<< KEY FIX

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=PAGE_SIZE,
        leftMargin=MARGINS[0],
        rightMargin=MARGINS[1],
        topMargin=MARGINS[2],
        bottomMargin=MARGINS[3],
        title=title_text,
        author="UC Online",
    )
    base, title, h2, h3, meta, _, _ = _styles()
    story: List[Flowable] = []

    # Cover page header.
    story.append(Paragraph(_strip_accents_if_needed(title_text), title))
    when = datetime.now().strftime("%d %b %Y, %I:%M %p")
    story.append(Paragraph(_strip_accents_if_needed(f"Generated: {when}"), meta))
    if degree_list:
        disp = ", ".join([DEGREE_LABELS.get(d, d) for d in degree_list])
        story.append(Paragraph(_strip_accents_if_needed(f"Degrees selected: <b>{disp}</b>"), base["Normal"]))
    story.append(Spacer(1, 0.25*inch))

    # Quick primer section.
    story.append(Paragraph("How to read this document", h2))
    story.append(Paragraph(
        _strip_accents_if_needed(
            "The ranking indicates relative development value, not a strict build sequence. "
            "Courses in the course list are relevant to degree completion (core or elective) under the current data. "
            "Use the following pages to review the top 20 courses at 100-, 200-, and 300-levels for each major."
        ),
        base["Normal"]
    ))
    story.append(Spacer(1, 0.12*inch))

    # Summary of the selection.
    story.append(Paragraph("Summary", h2))
    story.append(Paragraph("Target degrees with corresponding Majors", h3))
    story.append(_degrees_with_majors_bullets(degree_list, plan, majors_map))
    story.append(Spacer(1, 0.12*inch))

    # Assumptions section.
    story.append(Paragraph("Assumptions", h2))
    story.append(Paragraph(
        _strip_accents_if_needed(
            "- Optimisation uses enrolments, student satisfaction, and overlap between degrees; "
            "other institutional constraints (e.g., cost, staffing) are not modelled here.<br/>"
            "- Recommendations are advisory; final decisions remain with UC Online.<br/>"
            "- Input course data is current and pre-cleaned (no duplicates, consistent naming)."
        ),
        base["Normal"]
    ))

    # Start degree/major pages after the front-matter.
    story.append(PageBreak())

    # Degree pages (one major per page; respect selected majors).
    for deg in degree_list:
        sub_degree = plan[plan["Degree"] == deg].copy()

        # Use selected majors if provided for this degree; else fall back to all majors in plan.
        if deg in majors_map and majors_map[deg]:
            majors = majors_map[deg]
        else:
            majors = (
                sub_degree["Primary Major"].dropna().astype(str).sort_values().unique().tolist()
                if "Primary Major" in sub_degree.columns else []
            )

        if not majors:
            padded = _pad_min_per_level(sub_degree, plan, deg, min_per_level=min_per_level)
            blocks = _level_table_blocks(padded)
            if blocks:
                story.append(Paragraph(_strip_accents_if_needed(f"{DEGREE_LABELS.get(deg, deg)}"), h2))
                for block in blocks:
                    story.extend(block)
                story.append(PageBreak())
            continue

        for mj in majors:
            sub_major = sub_degree[sub_degree.get("Primary Major", "") == mj].copy()
            padded = _pad_min_per_level(sub_major, plan, deg, min_per_level=min_per_level)
            blocks = _level_table_blocks(padded)
            if blocks:
                story.append(Paragraph(_strip_accents_if_needed(f"{DEGREE_LABELS.get(deg, deg)} — {mj}"), h2))
                for block in blocks:
                    story.extend(block)
                story.append(PageBreak())  # one major per page

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf.read()
