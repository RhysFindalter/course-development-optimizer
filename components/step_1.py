import streamlit as st
from utils.state import on_degree_change as state_on_degree_change, clear_degrees

DEGREE_LABEL_TO_CODE = {
    "Bachelor of Arts (BA)": "BA",
    "Bachelor of Health (BHlth)": "BHlth",
    "Bachelor of Criminal Justice (BCJ)": "BCJ",
}


def _handle_degree_change():
    """Keep the degree selection tidy before handing off to shared callbacks."""
    selected = st.session_state.get("degrees", [])
    valid = [lbl for lbl in selected if lbl in DEGREE_LABEL_TO_CODE]
    st.session_state["degrees"] = valid
    state_on_degree_change()

def render():
    st.session_state.setdefault("degrees", [])
    st.session_state.setdefault("degrees_prev", [])

    with st.container(border=True):
        st.subheader("1. Select Your Target Degrees")
        st.markdown(
            """<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-6">
  <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
</svg>
Select <b>all</b> degrees you intend to develop. This ensures the tool can generate an accurate course list.""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

        chosen_degrees = st.multiselect(
            "Select degree(s)",
            options=list(DEGREE_LABEL_TO_CODE.keys()),
            key="degrees",
            on_change=_handle_degree_change,
        )

        chosen_deg_codes = [DEGREE_LABEL_TO_CODE[lbl] for lbl in chosen_degrees]

        if chosen_deg_codes:
            st.caption(f"Selected degrees: {', '.join(chosen_deg_codes)}")
            st.button("Clear All", on_click=clear_degrees)
