"""
FAL Dashboard — main entry point.

Run with:  streamlit run app.py

Uses st.navigation / st.Page (Streamlit >= 1.36) to build a sidebar
navigation bar across multiple pages. Add/remove pages by editing the
`pages/` list below.
"""

import streamlit as st

st.set_page_config(
    page_title="Fantasy Anime League Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Page registry — each entry points to a script in pages/
# ---------------------------------------------------------------------------
kpi_page = st.Page(
    "pages/1_KPI_Dashboard.py",
    title="KPI Dashboard",
    icon="🎯",
    default=True,
)

explorer_page = st.Page(
    "pages/2_Data_Explorer.py",
    title="Data Explorer",
    icon="📊",
)

about_page = st.Page(
    "pages/3_About.py",
    title="About / Notes",
    icon="ℹ️",
)

nav = st.navigation(
    {
        "Overview": [kpi_page],
        "Analysis": [explorer_page],
        "Info": [about_page],
    }
)

# Sidebar branding, shown above the nav-generated page links on every page
with st.sidebar:
    st.markdown("## 📺 FAL Dashboard")
    st.caption("Fantasy Anime League — Spring 2026")
    st.divider()

nav.run()