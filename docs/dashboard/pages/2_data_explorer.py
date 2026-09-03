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
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
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
            st.scatter_chart(chart_df, x=x_col, y=y_col, color=color_col, use_container_width=True)
        else:
            st.scatter_chart(chart_df, x=x_col, y=y_col, use_container_width=True)

with tab_summary:
    st.dataframe(filtered_df.describe(include="all").transpose(), use_container_width=True)

    if "genres" in filtered_df.columns:
        st.subheader("Genre frequency (filtered view)")
        genre_counts = filtered_df["genres"].explode().value_counts()
        st.bar_chart(genre_counts)