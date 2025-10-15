from pathlib import Path
import streamlit as st

def load_css(filename: str = "style.css"):
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "assets" / filename,
        here.parent.parent / "assets" / filename,
        Path.cwd() / "assets" / filename,
    ]
    for p in candidates:
        if p.exists():
            st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
            return
    st.warning("assets/style.css not found near app or pages/.")
