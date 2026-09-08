# main/forecast_graphing.py
# Takes the raw json, turn them into proper scores using get_measures, and plot that against predicted scores from the Kalman filter.
import json
from pathlib import Path

import matplotlib.pyplot as plt


def graph(measured_scores, predicted_scores):
    """
    measured_scores: {title: [day0, day1, ..., dayN]}      -- starts at day 0
    predicted_scores: {title: [day1, day2, ..., dayN]}     -- starts at day 1 (one shorter)
    """
    num_days = len(next(iter(measured_scores.values())))
    measured_days = list(range(num_days))

    plt.figure(figsize=(10, 6))

    for title in measured_scores:
        (line,) = plt.plot(
            measured_days, measured_scores[title], label=f"{title} (Measured)", linestyle="-"
        )
        color = line.get_color()

        title_predicted = predicted_scores.get(title, [])
        if title_predicted:
            # predicted_scores starts at day 1, so offset the x-axis by 1
            # and size it to match however many predicted points we have.
            predicted_days = list(range(1, 1 + len(title_predicted)))
            plt.plot(
                predicted_days,
                title_predicted,
                label=f"{title} (Predicted)",
                linestyle="--",
                color=color,
            )

    plt.xlabel("Day")
    plt.ylabel("Score")
    plt.title("Measured vs. Predicted Scores")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("data/kalman_predictions/graphs/forecasting.png")


def get_measures(current_json, current_criteria):
    final_scores = {}

    for title in current_json[0]:
        final_scores[title] = [0 for _ in range(len(current_json))] # number of days
        for day in current_json:
            main_dict = current_json.get(day, {})
            criteria_dict = current_criteria.get(day, {})
            for metric in ['score', 'wc', 'favorites', 'dropped', 'forum']:
                final_scores[title][day] += main_dict.get(title, {}).get(metric, 0) * criteria_dict.get(title, {}).get(metric, 0)

    return final_scores


if __name__ == "__main__":
    # Variables:
    # measured_scores: JSON and should have a day zero, updated by get_measures()
    # current_json: updated from update_json() , has a day zero
    # current_criteria: to be multiplied to the stats in current_json, has a day zero, manually typed from weekly results
    # (thank god it's weekly)
    # predicted_scores: JSON, should only start from day one, manually typed from forecasting.py
    # Structures:
    # measured_scores = {
    #     "Insert title here" : [day0_score, day1_score, etc]
    # }
    # current_json = {
    #     0 : {
    #          "Insert title here" : {
    #               metric1: num
    #               metric2:num
    #          }
    #     }
    #     1 : {
    #          "Insert title here" : {
    #               metric1: num
    #               metric2:num
    #          }
    #     }
    #     ...
    # }
    # predicted_scores same as current_json but one-indexed
    # current_criteria = {
    #     0 : {
    #          "Insert title here" : {
    #               metric1 : num
    #               metric2: num
    #          }
    #     }
    #     ...
    # }


    target_dir = Path("data/kalman_predictions")
    current_path = target_dir / "current_json.json"
    criteria_path = target_dir / "current_criteria.json"
    predicted_path = target_dir / "predicted_scores.json"

    def load_json(path):
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file at {path} does not exist.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file at {path} contains invalid JSON formatting.")
            return {}

    current_json = load_json(current_path)
    current_criteria = load_json(criteria_path)
    predicted_scores = load_json(predicted_path)

    current_json = {int(k): v for k, v in current_json.items()}
    current_criteria = {int(k): v for k, v in current_criteria.items()}

    if not current_json:
        raise SystemExit(
            f"No data found at {current_path}. Populate current_json.json "
            "(day -> title -> metric dict, matching update_json.py's output) before running this script."
        )

    # updating measured_scores:
    measured_scores = get_measures(current_json, current_criteria)

    graph(measured_scores, predicted_scores)