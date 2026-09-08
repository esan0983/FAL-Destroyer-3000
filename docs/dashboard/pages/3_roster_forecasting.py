# pages/forecast_view.py
# Interactive Streamlit view of measured vs. predicted FAL scores for the roster.
# Skeleton: wires up data loading + roster checkboxes + plotting. Fill in TODOs
# once real data paths / roster source are finalized.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

# Reuses the already-tested get_measures() from main/forecast_graphing.py.
# Adjust the import path to match wherever this actually lives relative to
# pages/ once it's wired into the real app (e.g. `from main.forecast_graphing
# import get_measures` if main/ is a package).
from main.forecast_graphing import get_measures

DATA_DIR = Path("data/kalman_predictions")


# ---------------------------------------------------------------------------
# Data loading (cached so re-running the script on checkbox clicks doesn't
# re-read JSON off disk every time)
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


@st.cache_data
def load_all_data():
    current_json = load_json(DATA_DIR / "current_json.json")
    current_criteria = load_json(DATA_DIR / "current_criteria.json")
    predicted_scores = load_json(DATA_DIR / "predicted_scores.json")

    # JSON day keys come back as strings; get_measures indexes with ints.
    current_json = {int(k): v for k, v in current_json.items()}
    current_criteria = {int(k): v for k, v in current_criteria.items()}

    return current_json, current_criteria, predicted_scores


# ---------------------------------------------------------------------------
# Plotting (adapted from forecast_graphing.graph(), but only draws the
# titles the user has checked, and skips predicted lines cleanly when a
# title has none)
# ---------------------------------------------------------------------------
def plot_selected(measured_scores, predicted_scores, selected_titles, show_predicted):
    if not selected_titles:
        st.info("Select at least one anime from the roster to see its forecast.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for title in selected_titles:
        if title not in measured_scores:
            continue

        measured = measured_scores[title]
        measured_days = list(range(len(measured)))
        (line,) = ax.plot(measured_days, measured, label=f"{title} (Measured)", linestyle="-")
        color = line.get_color()

        if show_predicted:
            predicted = predicted_scores.get(title, [])
            if predicted:
                predicted_days = list(range(1, 1 + len(predicted)))
                ax.plot(
                    predicted_days,
                    predicted,
                    label=f"{title} (Predicted)",
                    linestyle="--",
                    color=color,
                )

    ax.set_xlabel("Day")
    ax.set_ylabel("Score")
    ax.set_title("Measured vs. Predicted Scores")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main():
    st.title("FAL Forecast Viewer")
    st.caption("Measured vs. Kalman-filter-predicted scores for your roster.")

    current_json, current_criteria, predicted_scores = load_all_data()

    if not current_json:
        st.warning(
            f"No data found in `{DATA_DIR}`. Run the data collection / "
            "Kalman filter pipeline first, or check the file paths above."
        )
        return

    measured_scores = get_measures(current_json, current_criteria)
    all_titles = sorted(measured_scores.keys())

    # TODO: replace this with the actual roster source (e.g. a saved roster
    # list / session state from a team-setup page) instead of every title
    # in current_json.
    roster_titles = all_titles

    with st.sidebar:
        st.subheader("Roster")
        show_predicted = st.checkbox("Show predicted scores", value=True)

        # TODO: swap for st.multiselect if the roster gets long — checkboxes
        # are nice for a small 5-8 anime roster but get unwieldy past that.
        selected_titles = [
            title for title in roster_titles if st.checkbox(title, value=True, key=f"cb_{title}")
        ]

    plot_selected(measured_scores, predicted_scores, selected_titles, show_predicted)

    # TODO: KPI cards row (current criteria-weighted score per selected
    # anime, rank vs. rest of roster, etc.) to match the rest of the
    # dashboard's KPI-card style.


if __name__ == "__main__":
    main()