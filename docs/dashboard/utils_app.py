"""
Shared helpers used across pages: cached data loading + a reusable
top-of-page filter bar.

Keeping this separate avoids duplicating parquet-loading / filter logic
in every page file.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import duckdb

STATS_PARQUET_PATH = Path("data/processed/stats_df.parquet")


@st.cache_data(show_spinner="Loading stats_df.parquet...")
def load_stats_df(path: str = str(STATS_PARQUET_PATH)) -> pd.DataFrame:
    """Load the cohort-level stats dataframe produced by feature_engineering.pre_split.

    Cached so repeated page interactions (filter changes, reruns) don't
    re-read the parquet file from disk every time.
    """
    p = Path(path)
    if not p.exists():
        st.error(
            f"Could not find `{p}`. Make sure feature_engineering.py has been "
            "run and stats_df.parquet exists at that path."
        )
        return pd.DataFrame()
    return pd.read_parquet(p, engine="pyarrow")

@st.cache_data
def get_column_bounds(df):
    """Calculates min/max once and caches the result."""
    bounds = {}
    for col in df.select_dtypes(include="number").columns:
        if df[col].notna().any():
            bounds[col] = (float(df[col].min()), float(df[col].max()))
    return bounds

def render_top_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Render a row of filter widgets at the top of a page and return the
    filtered dataframe. Filters degrade gracefully if a column is missing.

    key_prefix keeps widget keys unique across pages so Streamlit doesn't
    collide state between, e.g., the KPI page and the Data Explorer page.
    """
    if df.empty:
        return df

    with st.container(border=True):
        st.markdown("**Filters**")
        cols = st.columns(5)
        filtered = df.copy()
        bounds = get_column_bounds(df)

        # Rating filter
        with cols[0]:
            if "rating" in df.columns:
                options = sorted(df["rating"].dropna().unique().tolist())
                selected = st.multiselect(
                    "Rating", options, default=[], key=f"{key_prefix}_rating"
                )
                if selected:
                    filtered = filtered[filtered["rating"].isin(selected)]
            else:
                st.caption("No `rating` column")

        # Source filter
        with cols[1]:
            if "source" in df.columns:
                options = sorted(df["source"].dropna().unique().tolist())
                selected = st.multiselect(
                    "Source", options, default=[], key=f"{key_prefix}_source"
                )
                if selected:
                    filtered = filtered[filtered["source"].isin(selected)]
            else:
                st.caption("No `source` column")

        # Sequel filter
        with cols[2]:
            if "sequel" in df.columns:
                choice = st.selectbox(
                    "Sequel status",
                    ["All", "Sequel only", "Original only"],
                    key=f"{key_prefix}_sequel",
                )
                if choice == "Sequel only":
                    filtered = filtered[filtered["sequel"].astype(bool)]
                elif choice == "Original only":
                    filtered = filtered[~filtered["sequel"].astype(bool)]
            else:
                st.caption("No `sequel` column")

        # Score range filter
        with cols[3]:
            if "score" in df.columns and df["score"].notna().any():
                if "score" in bounds:
                    lo, hi = bounds["score"]
                    if lo < hi:
                        z_range = st.slider(
                            "score range",
                            min_value=lo,
                            max_value=hi,
                            value=(lo, hi),
                            key=f"{key_prefix}_score",
                        )
                        filtered = filtered[
                            filtered["score"].between(z_range[0], z_range[1])
                        ]
                else:
                    st.caption("Score does not exist in bounds")
            else:
                st.caption("No `score` column")

        # Score range filter
        with cols[4]:
            if "drop_rate" in df.columns and df["drop_rate"].notna().any():
                if "drop_rate" in bounds:
                    lo, hi = bounds["drop_rate"]
                    if lo < hi:
                        z_range = st.slider(
                            "drop_rate range",
                            min_value=lo,
                            max_value=hi,
                            value=(lo, hi),
                            key=f"{key_prefix}_drop_rate",
                        )
                        filtered = filtered[
                            filtered["drop_rate"].between(z_range[0], z_range[1])
                        ]
                else:
                    st.caption("Drop rate does not exist in bounds")
            else:
                st.caption("No `drop_rate` column")

        st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")

    return filtered