# main/roster_selection.py
# Uses prediction data to create a roster by z-scoring and putting the z-scores into a heuristic formula

import json
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint

# Takes the JSON values and turns them into z-scores
# Data is approximately normal for non-score metrics since they were logged
def z_scores(target_data):
    nums = list(target_data.values())
    mean = np.mean(nums)
    std = np.std(nums)

    dict_z = {key: (value - mean) / std for key, value in target_data.items()}

    return dict_z

# Turns z-scored data into a weighted score per target
def heuristic(data, target):
    criteria = {
        'score': 2/9,
        'wc': 5/9,
        'favorites': 1/6,
        'dropped': -1/9,
        'forum': 1/6
    }

    multiplier = criteria.get(target, 1_000_000)
    if multiplier == 1_000_000:
        raise ValueError("Resorted to sentinel value")

    dict_heuristic = {key: value * multiplier for key, value in data.items()}

    return dict_heuristic

# This is where all the score processing happens
# Prints and saves final roster
def roster(titles):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    directory_path = Path("data/ml_predictions/roster")

    final_dict = {title: 0 for title in titles}

    for metric in metrics:
        file_path = directory_path / f"{metric}_predictions.json"
    
        try:
            with file_path.open("r", encoding="utf-8") as file:
                target_data = json.load(file)
        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except json.JSONDecodeError:
            print("Failed to decode JSON. Check if the file format is valid.")

        target_data = z_scores(target_data)
        target_data = heuristic(target_data, metric)
        for key in final_dict:
            value = target_data.get(key, 1_000_000)
            if value == 1_000_000:
                raise ValueError("Resorted to sentinel value")
            final_dict[key] += value


    final_dict = dict(sorted(final_dict.items(), key=lambda item: item[1], reverse=True))
    pprint(final_dict, indent=4, sort_dicts=False)

    file_path = directory_path / "roster.json"

    with open(file_path, "w") as json_file:
        json.dump(final_dict, json_file, indent=4)
    
if __name__ == "__main__":
    df = pd.read_csv("data/raw/current_data.csv")
    titles = df['title']
    roster(titles)