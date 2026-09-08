# main/fall_stats.py
# Takes the standard deviation of all the 73 predictions for each metric
# This is done so we can scale the RMSE for the noise matrix of the Kalman filter.
from pathlib import Path
import numpy as np
from pprint import pprint

def get_std():
    json_dict = {}
    import json
    std_dict = {}

    for metric in ['score', 'wc', 'favorites', 'dropped', 'forum']:
        target_dir = Path("data/ml_predictions/roster")
        file_path = target_dir / f"{metric}_predictions.json"
        json_dict[metric] = {}
        try:
            with file_path.open("r", encoding="utf-8") as file:
                json_dict[metric] = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file at {file_path} does not exist.")
        except json.JSONDecodeError:
            print(f"Error: The file at {file_path} contains invalid JSON formatting.")

        std_dict[metric] = np.std(list(json_dict.get(metric, {}).values()))

    print("Standard deviations:")
    pprint(std_dict, indent=4)

if __name__ == "__main__":
    get_std()

        