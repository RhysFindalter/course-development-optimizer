import streamlit as st

def wipe_plan():
    """Remove persisted plan + degrees so Step 3 asks to regenerate."""
    st.session_state.pop("course_plan", None)
    st.session_state.pop("course_plan_degrees", None)

def on_degree_change():
    """Callback when multiselect changes; wipe plan if selection changed."""
    current = set(st.session_state.get("degrees", []))
    prev = set(st.session_state.get("degrees_prev", []))
    if current != prev:
        wipe_plan()
    st.session_state["degrees_prev"] = list(current)

def clear_degrees():
    """Callback to clear selection safely (allowed for widget-backed keys)."""
    st.session_state["degrees"] = []
    st.session_state["degrees_prev"] = []
    st.session_state["clear_all_clicked"] = True  # flag for other steps
    wipe_plan()
