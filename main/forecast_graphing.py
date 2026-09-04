import json
import matplotlib.pyplot as plt

# untested
def graph(measured_scores, predicted_scores):
    num_days = len(next(iter(measured_scores.values())))
    days = list(range(num_days))

    plt.figure(figsize=(10, 6))

    for title in measured_scores:
        (line,) = plt.plot(
            days, measured_scores[title], label=f"{title} (Measured)", linestyle="-"
        )
        color = line.get_color()

        plt.plot(
            days,
            predicted_scores[title],
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

# untested
def get_measures(current_json, current_criteria):
    final_scores = {}
    for day in current_json:
        final_scores[day] = {}
        main_dict = current_json.get(day, {})
        criteria_dict = current_criteria.get(day, {})

        for title in main_dict:
            final_scores[day][title] = 0

            for metric in ['score', 'wc', 'favorites', 'dropped', 'forum']:
                final_scores[title] += main_dict.get(title, {}).get(metric, 0) * criteria_dict.get(title, {}).get(metric, 0)

    return final_scores


if __name__ == "__main__":
    # Variables:
    # measured_scores: JSON and should have a day zero, updated by get_measures()
    # current_json: updated from update_json() 
    # current_criteria: to be multiplied to the stats in current_json
    # predicted_scores: JSON, should only start from day one
    # Both JSONs should have this structure:
    # measured_scores = {
    #     "Insert title here" : [day1_score, day2_score, etc]
    # }

    predictions, current_criteria, predicted_scores = {}, {}, {}

    # updating measured_scores:
    measured_scores = get_measures(predictions, current_criteria)

    graph(measured_scores, predicted_scores)
    