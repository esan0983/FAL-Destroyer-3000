import json
import matplotlib.pyplot as plt

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
    plt.savefig("data/ml_predictions/graphs/forecasting.png")

if __name__ == "__main__":
    # Variables:
    # measured_scores: JSON and should have a day zero
    # predicted_scores: JSON, should only start from day one
    # Both JSONs should have this structure:
    # measured_scores = {
    #     "Insert title here" : [day1_score, day2_score, etc]
    # }
    pass