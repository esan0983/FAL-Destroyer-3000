import streamlit as st

st.title("ℹ️ About")

st.markdown(
    """
This is a starter template for the FAL project's Streamlit dashboard.

**Structure**

- `app.py` — entry point; declares the sidebar navigation via `st.navigation`.
- `utils_app.py` — shared cached data loading + a reusable top-of-page filter bar.
- `pages/1_KPI_Dashboard.py` — manual KPI inputs (targets, thresholds) with
  derived summary metrics.
- `pages/2_Data_Explorer.py` — browses `data/processed/stats_df.parquet`
  with table, chart, and summary-stats views.
- `pages/3_About.py` — this page.

**Extending it**

- Add a new page file under `pages/`, then register it in `app.py`'s
  `st.navigation(...)` call.
- Add new filters in `utils_app.render_top_filters` — it's shared across
  pages, so a new filter shows up everywhere automatically.
- Swap `load_stats_df` for other parquet/csv sources by adding sibling
  `load_*` functions in `utils_app.py`, following the same `st.cache_data`
  pattern.
"""
)