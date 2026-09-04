"""
KPI Dashboard page.

A non-technical, at-a-glance summary of the (filtered) catalog: total
anime count, number of distinct genres, average score, etc. — the kind of
numbers you'd put in front of a stakeholder who doesn't care about
score_z or SVD components.

The "inputs" here are which KPI cards to show and a couple of thresholds
that feed the derived cards (e.g. "what counts as a high score"), rather
than technical model parameters.
"""

import streamlit as st

from utils_app import load_stats_df, render_top_filters

st.title("🎯 KPI Dashboard")
# st.caption("A plain-language snapshot of the catalog for the current filter selection.")

df = load_stats_df()

# ---------------------------------------------------------------------------
# KPI configuration — what to show, and thresholds that feed derived KPIs
# ---------------------------------------------------------------------------
st.subheader("KPI Settings")

with st.form("kpi_settings_form"):
    st.markdown("**Which KPIs should be shown?**")
    kpi_choices = st.multiselect(
        "KPI cards",
        [
            "Total anime",
            "Unique genres",
            "Unique studios",
            "Unique producers",
            "Average score",
            "% sequels",
            "% high scoring",
            "% high drop rate",
        ],
        default=[
            "Total anime",
            "Unique genres",
            "Unique studios",
            "Average score",
            "% sequels",
            "% high scoring",
        ],
    )

    st.markdown("**Thresholds** (used by the derived KPIs above)")
    c1, c2 = st.columns(2)
    with c1:
        high_score_cutoff = st.number_input(
            "High score cutoff (a title counts as 'high scoring' at or above this)",
            value=1.0,
            step=0.1,
            format="%.2f",
        )
    with c2:
        high_drop_cutoff = st.number_input(
            "High drop_rate cutoff (a title counts as 'high drop rate' at or above this)",
            value=1.0,
            step=0.1,
            format="%.2f",
        )

    submitted = st.form_submit_button("Update KPIs", use_container_width=True)

if submitted:
    st.session_state["kpi_settings"] = {
        "kpi_choices": kpi_choices,
        "high_score_cutoff": high_score_cutoff,
        "high_drop_cutoff": high_drop_cutoff,
    }

settings = st.session_state.get(
    "kpi_settings",
    {
        "kpi_choices": [
            "Total anime",
            "Unique genres",
            "Unique studios",
            "Average score",
            "% sequels",
            "% high scoring",
        ],
        "high_score_cutoff": 1.0,
        "high_drop_cutoff": 1.0,
    },
)

st.divider()

# ---------------------------------------------------------------------------
# Compute KPI values
# ---------------------------------------------------------------------------
n_total = len(df)


def n_unique_exploded(col):
    if col not in df.columns:
        return None
    return df[col].explode().nunique(dropna=True)


kpi_values = {}

kpi_values["Total anime"] = (f"{n_total:,}", None)

genres_unique = n_unique_exploded("genres")
kpi_values["Unique genres"] = (
    f"{genres_unique:,}" if genres_unique is not None else "N/A",
    None,
)

studios_unique = n_unique_exploded("studios")
kpi_values["Unique studios"] = (
    f"{studios_unique:,}" if studios_unique is not None else "N/A",
    None,
)

producers_unique = n_unique_exploded("producers")
kpi_values["Unique producers"] = (
    f"{producers_unique:,}" if producers_unique is not None else "N/A",
    None,
)

if "score" in df.columns and df["score"].notna().any():
    avg_score = df["score"].mean()
    kpi_values["Average score"] = (f"{avg_score:+.2f}", None)
else:
    kpi_values["Average score"] = ("N/A", None)

if "sequel" in df.columns and n_total:
    pct_sequel = df["sequel"].astype(bool).mean() * 100
    kpi_values["% sequels"] = (f"{pct_sequel:.1f}%", None)
else:
    kpi_values["% sequels"] = ("N/A", None)

if "score" in df.columns and n_total:
    n_high_score = (df["score"] >= settings["high_score_cutoff"]).sum()
    pct_high_score = n_high_score / n_total * 100
    kpi_values["% high scoring"] = (
        f"{pct_high_score:.1f}%",
        f"{n_high_score:,} titles >= {settings['high_score_cutoff']:.2f}",
    )
else:
    kpi_values["% high scoring"] = ("N/A", None)

if "drop_rate" in df.columns and n_total:
    n_high_drop = (df["drop_rate"] >= settings["high_drop_cutoff"]).sum()
    pct_high_drop = n_high_drop / n_total * 100
    kpi_values["% high drop rate"] = (
        f"{pct_high_drop:.1f}%",
        f"{n_high_drop:,} titles >= {settings['high_drop_cutoff']:.2f}",
    )
else:
    kpi_values["% high drop rate"] = ("N/A", None)

# ---------------------------------------------------------------------------
# Render KPI cards, 4 per row
# ---------------------------------------------------------------------------
st.subheader("Summary")

selected = [k for k in settings["kpi_choices"] if k in kpi_values]

if not selected:
    st.info("Pick at least one KPI card above and click **Update KPIs**.")
else:
    rows = [selected[i : i + 4] for i in range(0, len(selected), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, label in zip(cols, row):
            value, helptext = kpi_values[label]
            col.metric(label, value, help=helptext)

st.caption(f"Based on {n_total:,} titles matching the current filters.")