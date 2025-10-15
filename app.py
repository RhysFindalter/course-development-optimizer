import streamlit as st
import pandas as pd
import plotly.io as pio

from utils.styling import load_css
from config.settings import APP_TITLE
from services.db import health_check, fetch_courses_df_for_degrees
from components import step_1
from components import step_2
from components import step_3
from components import step_4
from components import step_5

# Page setup.
st.set_page_config(page_title=APP_TITLE, layout="wide")
load_css()
pio.templates.default = "plotly_white"

ok, msg = health_check()
if not ok:
    st.error(f"Database connection failed: {msg}")
    st.stop()

st.title("UC Online Course List Tool")
st.markdown(
    "<p class='subtext'>This tool recommends a course list of the most optimal courses to consider "
    "when developing degree-majors based on enrolments, satisfaction, and overlap across degrees.</p>",
    unsafe_allow_html=True,
)

# Step 1: collect degree selections and store them in session state.
step_1.render()

LABEL_TO_DB_CODE = {
    "Bachelor of Arts (BA)": "BA",
    "Bachelor of Health (BHlth)": "BHlth",
    "Bachelor of Health Sciences (BHSc)": "BHlth",
    "Bachelor of Criminal Justice (BCJ)": "BCJ",
    "BA": "BA", "BHSc": "BHlth", "BHlth": "BHlth", "BCJ": "BCJ",
}
raw_degree_labels = st.session_state.get("degrees", [])
selected_degree_codes = [LABEL_TO_DB_CODE.get(x, x) for x in raw_degree_labels]

# Pull the authoritative course data for the chosen degrees.
if selected_degree_codes:
    courses_df = fetch_courses_df_for_degrees(selected_degree_codes)
else:
    courses_df = pd.DataFrame(columns=[
        "Course Code", "Course Name", "Degree", "Major",
        "Enrolment Score", "Satisfaction Score", "Overlap Score", "Optimised Score"
    ])

if "optimised_score" in courses_df.columns and "Optimised Score" not in courses_df.columns:
    courses_df = courses_df.rename(columns={"optimised_score": "Optimised Score"})

# Fill the optimised score locally if the source omitted it.
if "Optimised Score" not in courses_df.columns:
    if all(c in courses_df.columns for c in ["Enrolment Score", "Satisfaction Score", "Overlap Score"]):
        courses_df["Optimised Score"] = (
            0.01 * courses_df["Enrolment Score"].fillna(0)
            + 2.0 * courses_df["Satisfaction Score"].fillna(0)
            + 1.0 * courses_df["Overlap Score"].fillna(0)
        ).round(2)
    else:
        courses_df["Optimised Score"] = pd.Series(dtype="float64")

st.session_state["courses_df"] = courses_df

# Step 2 surfaces majors for the active degree set.
step_2.render()

# Step 3 confirms the selection and kicks off optimisation.
step3_clicked = step_3.render()

if st.session_state.get("degree_codes"):
    degrees_for_steps = st.session_state["degree_codes"]
else:
    raw_labels = st.session_state.get("degrees", [])
    degrees_for_steps = [LABEL_TO_DB_CODE.get(x, x) for x in raw_labels]

majors_detail = st.session_state.get("majors_detail", [])
majors_df = pd.DataFrame(majors_detail) if majors_detail else pd.DataFrame([])

# Step 4 builds the ranked course plan.
step_4.render(st.session_state["courses_df"], clicked=step3_clicked, degrees=degrees_for_steps)

# Step 5 visualises the results and exports.
step_5.render(data=majors_df, degrees=degrees_for_steps)
