# Course Development Optimizer (CDO)

The **Course Development Optimizer** is an interactive web application built with **Streamlit** and **PostgreSQL**, designed to help course developers plan and optimize their degree-major developments. It allows visualization, scoring, and reporting of course combinations across multiple degrees.

---

## Features

- Interactive web UI powered by **Streamlit**
- Postg[text](vscode-webview://03gs87thknhmu2t1g3gc952ba4nfsq8577rrhgm74vqp2c16ifh7/login)reSQL database (hosted on [Neon](https://neon.com/))
- Course planning across multiple degrees (BA, BHSc, BCJ, etc.)
- Optimization scoring (Enrolment, Satisfaction, Overlap, Optimised Score)
- Visualization of course data using **Plotly**
- PDF report export via **ReportLab**
- Excel export via **openpyxl** / **xlsxwriter**

---

## Tech Stack

- **Programming Language:** Python 3.10
- **Database:** PostgreSQL (managed with pgAdmin 4, hosted on Neon)
- **Frontend/UI:** Streamlit
- **Data Handling:** pandas, numpy
- **Visualization:** plotly
- **Export:** openpyxl, xlsxwriter, reportlab
- **Database Driver:** psycopg2

---

## Project Structure

```
course-development-optimizer/
│── app.py                # Main Streamlit app entry point
│── components/           # UI components for each step (step_1.py ... step_5.py)
│── config/               # Settings and configuration files
│── services/             # Database services (PostgreSQL connection)
│── utils/                # Utility functions (charts, reports, styling, state)
```

---

## Installation

1. Clone or extract this repository:

   ```bash
   git clone <repo_url>
   cd course-development-optimizer
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser.

---

## Example Workflow

1. Select a degree (BA, BHSc, BCJ, etc.)
2. Generate course lists and optimization scores
3. Visualize results via scatterplots and tables
4. Export reports to **PDF** or **Excel**

---

## License

This project is for educational purposes as part of the **University of Canterbury Course Optimization project**.  
Not intended for production use without modification.
