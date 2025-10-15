import re
import streamlit as st


CANON_MAP = {
    "BJC": "BCJ", "BCJ": "BCJ",
    "BHSC": "BHlth", "BHSc": "BHlth", "BHS": "BHlth", "BHlth": "BHlth",
    "BA": "BA",
    "BPSYC": "BPsyc", "BPsyc": "BPsyc",
}

def _extract_code(token: str) -> str:
    s = str(token).strip()
    if not s:
        return ""
    s = s.split('(')[0].strip()
    s = s.split()[0] if s else ""
    s = re.sub(r"[^\w]", "", s)
    return s

def _normalise_degree_codes(selected: list[str]) -> list[str]:
    out, seen = [], set()
    for item in selected or []:
        raw = _extract_code(item)
        canon = CANON_MAP.get(raw, raw)
        if canon and canon not in seen:
            out.append(canon); seen.add(canon)
    return out

def render() -> bool:
    with st.container(border=True):
        st.subheader("3. Review Selection and Generate Course List")
        st.write("Confirm your selected degrees and corresponding majors, then generate the course list.")

        degrees_raw = st.session_state.get("degrees", [])
        degree_codes = st.session_state.get("degree_codes") or _normalise_degree_codes(degrees_raw)

        # User selections (majors actually chosen)
        majors_by_degree: dict[str, list[str]] = st.session_state.get("majors_by_degree", {})

        # Catalogue (majors that were OFFERED for each degree in Step 2)
        available_by_degree: dict[str, list[str]] = st.session_state.get("available_majors_by_degree", {})

        st.markdown(
            """
            <style>
              .pills { display:flex; flex-wrap:wrap; gap:.4rem; }
              .pill { padding:.25rem .6rem; border-radius:999px; border:1px solid rgba(0,0,0,.1); }
              .pill--major { font-size:0.85rem; opacity:.95; }
              .degree-card { border:1px solid rgba(0,0,0,.10); border-radius:.75rem; padding:.6rem .8rem; margin:.5rem 0; }
              .degree-header { display:flex; align-items:center; gap:.5rem; margin:0 0 .4rem 0; }
              .degree-title { font-weight:600; }
              .muted { color:rgba(0,0,0,.55); font-size:.9rem; }
              .degree-card.is-error { border-color:#e53935 !important; background:rgba(229,57,53,.03); }
              .degree-card.is-error .muted { color:#b71c1c !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Degrees that REQUIRE a major selection (i.e., they had at least 1 available)
        require_selection = []
        for d in degree_codes:
            avail = available_by_degree.get(d, []) or []
            chosen = majors_by_degree.get(d, []) or []
            if len(avail) > 0 and len(chosen) == 0:
                require_selection.append(d)

        # Valid if: at least one degree AND no degrees that require selection are missing a selection
        is_valid = bool(degree_codes) and (len(require_selection) == 0)

        # Reset sticky errors if selection changed
        sig = (
            tuple(sorted(degree_codes)),
            tuple((d, tuple(sorted(majors_by_degree.get(d, []) or []))) for d in sorted(degree_codes)),
            tuple((d, tuple(sorted(available_by_degree.get(d, []) or []))) for d in sorted(degree_codes)),
        )
        if sig != st.session_state.get("_step3_sig"):
            st.session_state["_step3_sig"] = sig
            st.session_state["_step3_attempted"] = False

        attempted = bool(st.session_state.get("_step3_attempted", False))

        # Cards
        cards_ph = st.empty()
        def render_cards(show_errors: bool):
            with cards_ph.container():
                if degree_codes:
                    for d in degree_codes:
                        avail = available_by_degree.get(d, []) or []
                        chosen = majors_by_degree.get(d, []) or []

                        has_avail = len(avail) > 0
                        needs_choice = has_avail and len(chosen) == 0

                        # Row styling: only flag as error if majors exist but none chosen
                        card_has_error = show_errors and needs_choice

                        if not has_avail:
                            # No majors exist for this degree
                            majors_html = "<span class='muted'>(no majors available)</span>"
                            count_text = "0 majors available"
                        else:
                            if chosen:
                                majors_html = "".join(f"<span class='pill pill--major'>{m}</span>" for m in chosen)
                            else:
                                majors_html = "<span class='muted'>(no majors selected)</span>"
                            count_text = f"{len(chosen)} selected / {len(avail)} available"

                        st.markdown(
                            f"""
                            <div class="degree-card{' is-error' if card_has_error else ''}">
                              <div class="degree-header">
                                <span class="degree-title">{d}</span>
                                <span class="muted">• {count_text}</span>
                              </div>
                              <div class="pills">{majors_html}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    if show_errors:
                        st.info("Select at least one **degree** in Step 1.")
        render_cards(show_errors=attempted)

        # Keep a dedicated spot for validation feedback above the button.
        msg_ph = st.empty()
        if attempted and not is_valid:
            with msg_ph.container():
                if not degree_codes:
                    st.info("Select at least one **degree** in Step 1.")
                else:
                    st.info("Select at least one **major** for: " + ", ".join(require_selection))

        # Submit button lives in a form so Streamlit handles the layout cleanly.
        with st.form("step3_action_form", border=False):
            submitted = st.form_submit_button(
                "Generate Course List",
                type="primary",
                width="stretch"
            )

            if submitted:
                if is_valid:
                    st.session_state["_step3_attempted"] = False
                    return True
                else:
                    # Remember the failed attempt so the card highlights stay visible.
                    st.session_state["_step3_attempted"] = True
                    render_cards(show_errors=True)
                    with msg_ph.container():
                        if not degree_codes:
                            st.info("Select at least one **degree** in Step 1.")
                        else:
                            st.info("Select at least one **major** for: " + ", ".join(require_selection))
                    return False

        return False
