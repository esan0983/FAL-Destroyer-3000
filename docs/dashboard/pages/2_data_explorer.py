"""
Data Explorer page.

Loads data/processed/stats_df.parquet (the pre-split, cohort-level dataframe
produced by feature_engineering.pre_split) and lets the user browse and
chart it interactively.
"""

import pandas as pd
import streamlit as st

from utils_app import load_stats_df, render_top_filters

st.title("📊 Data Explorer")
st.caption("Source: `data/processed/stats_df.parquet`")

df = load_stats_df()

if df.empty:
    st.stop()

# ---------------------------------------------------------------------------
# Top filter bar
# ---------------------------------------------------------------------------
filtered_df = render_top_filters(df, key_prefix="explorer")

st.divider()

# ---------------------------------------------------------------------------
# Tabs: table view / chart view / summary stats
# ---------------------------------------------------------------------------
tab_table, tab_chart, tab_summary = st.tabs(["Table", "Chart", "Summary Stats"])

with tab_table:
    # Initialize display limit state
    if "row_limit" not in st.session_state:
        st.session_state.row_limit = 100

    total_rows = len(filtered_df)
    current_limit = st.session_state.row_limit

    # Slice table for fast rendering
    displayed_df = filtered_df.head(current_limit)

    # Status indicator
    st.caption(f"Showing **{len(displayed_df):,}** of **{total_rows:,}** filtered rows")

    # Fast display of sliced dataset
    st.dataframe(displayed_df, width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # Pagination Controls
    # -----------------------------------------------------------------------
    if current_limit < total_rows:
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("Load next 100 rows"):
                st.session_state.row_limit += 100
                st.rerun()
        with col2:
            if st.button("Show all rows"):
                st.session_state.row_limit = total_rows
                st.rerun()
    elif total_rows > 100:
        if st.button("Reset view to top 100"):
            st.session_state.row_limit = 100
            st.rerun()

    st.divider()

    # Note: CSV Download gets the FULL filtered dataset, not just the sliced view
    st.download_button(
        "Download full filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="stats_df_filtered.csv",
        mime="text/csv",
    )

with tab_chart:
    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns to chart.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X axis", numeric_cols, index=0)
        with c2:
            default_y = 1 if len(numeric_cols) > 1 else 0
            y_col = st.selectbox("Y axis", numeric_cols, index=default_y)
        with c3:
            color_options = [None] + [
                c for c in filtered_df.columns if filtered_df[c].dtype == "object" or filtered_df[c].dtype.name == "category"
            ]
            color_col = st.selectbox("Color by (optional)", color_options, index=0)

        chart_df = filtered_df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()

        if color_col:
            st.scatter_chart(chart_df, x=x_col, y=y_col, color=color_col, width='stretch')
        else:
            st.scatter_chart(chart_df, x=x_col, y=y_col, width='stretch')

with tab_summary:
    st.dataframe(filtered_df.describe(include="all").transpose(), width='stretch')

    if "genres" in filtered_df.columns:
        st.subheader("Genre frequency (filtered view)")
        genre_counts = filtered_df["genres"].explode().value_counts()
        st.bar_chart(genre_counts)