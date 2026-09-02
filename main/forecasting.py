import numpy as np
import json
import pandas as pd
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt

# Kalman filter implementation
class KalmanFilter:
    def __init__(self, 
                 transition_mat,
                 observation_mat,
                 noise_mat,
                 measured_covs,
                 measurements):
        self.transition_mat = transition_mat # state transition matrix
        self.observation_mat = observation_mat # observation matrix, will most likely be I
        self.noise_mat = noise_mat # process noise
        self.measured_covs = measured_covs # covariance matrices across days; note that day 0 is a prediction from the neural network
        self.measurements = measurements # z variables in the documentation, 7 by n array

    # Check that inputs are properly shaped
    def check(self):
        # first check, transition matrix
        if self.transition_mat.shape[0] != self.transition_mat.shape[1]:
            return False
        else:
            dim = self.transition_mat.shape[0]

        # second check, observation matrix
        if self.observation_mat.shape[0] != self.observation_mat.shape[1] or self.observation_mat.shape[0] != dim:
            return False

        # third check, noise matrix
        if self.noise_mat.shape[0] != self.noise_mat.shape[1] or self.noise_mat.shape[0] != dim:
            return False

        # fourth check, measured covariance matrices
        measured_days = self.measurements.shape[0]
        if self.measured_covs.shape[0] != measured_days or self.measured_covs.shape[1] != self.measured_covs.shape[2] or self.measured_covs.shape[1] != dim:
            return False

        return True

    # Returns the next prediction
    def make_predictions(self):
        measured_days = self.measurements.shape[0] # starts with day 0
        num_features = self.measurements.shape[1]

        # more matrix initializations
        uncertainty_mats = np.zeros((measured_days + 1, measured_days, num_features, num_features)) # first two dimensions for the two subscripts, and then each entry is a two by two square matrix
        state_mats = np.zeros((measured_days + 1, measured_days, num_features)) # first two dimensions for the two subscripts, and the entry is a 1D array
        kalman_gains = np.zeros((measured_days, num_features, num_features))  # 1D array of 2 by 2 matrices

        state_mats[0][0] = self.measurements[0]
        
        for i in range(measured_days):
            uncertainty_mats[i][i] = self.measured_covs[i]

        # per-day updates
        for day in range(measured_days):
            state_mats[day+1][day] = self.transition_mat @ state_mats[day][day]

            uncertainty_mats[day+1][day] = self.transition_mat @ uncertainty_mats[day][day] @ self.transition_mat.T + self.noise_mat

            if day < measured_days - 1:
                kalman_gains[day+1] = uncertainty_mats[day+1][day] @ self.observation_mat @ np.linalg.inv(self.observation_mat @ uncertainty_mats[day+1][day] @ self.observation_mat.T + uncertainty_mats[day+1][day+1])

                state_mats[day+1][day+1] = state_mats[day+1][day] + kalman_gains[day+1] @ (self.measurements[day+1] - self.observation_mat @ state_mats[day+1][day])

        # return next day prediction
        return state_mats[measured_days][measured_days - 1]

# Receives measurements for all metrics and all days, and returns a list of covariance matrices
def sample_covariance(sample_json, num_features):
    covs = np.zeros((len(sample_json), num_features, num_features))

    for day_idx, day_dict in sample_json.items():

        cov_array = np.zeros((num_features, len(day_dict)))

        for anime_idx, (title, feature_dict) in enumerate(day_dict.items()):
            for feature_idx, feature_name in enumerate(feature_dict):
                cov_array[feature_idx, anime_idx] = feature_dict[feature_name]

        covs[day_idx] = np.cov(cov_array)

    return covs

# Predicts which anime you should ace
# If there is an anime that's within the intolerance range, I suggest holding off on acing
def acing_predictions(current_json, func_roster, ace_threshold, tolerance):
    roster = func_roster
    roster_wcs = {}
    current_stats = {}

    for day in current_json:
        current_stats[day] = {}

        for title in current_json.get(day, {}):
            current_stats[day][title] = dict(list(current_json.get(day, {}).get(title, {}).items())[-2:])

    for title in roster:
        roster_wcs[title] = np.zeros(2)
        temp_measurements = np.zeros((len(current_stats), 2))

        for day in current_stats:
            temp_measurements[day][0] = current_stats.get(day, {}).get(title, {}).get("wc_raw", 0)
            temp_measurements[day][1] = current_stats.get(day, {}).get(title, {}).get("wc_raw_dot", 0)

        temp_observation = np.eye(2)
        temp_noise = np.array([[6.25, 2.5], 
                                [2.5, 1]])
        temp_covs = sample_covariance(current_stats, 2)
        temp_transition = np.array([[1, 1], 
                                    [0, 1]])
        
        temp_kalman = KalmanFilter(temp_transition, temp_observation, temp_noise, temp_covs, temp_measurements)
        if not temp_kalman.check():
            print("Check failed for raw WC count!")
            return None
        roster_wcs[title] = temp_kalman.make_predictions()[0] # raw wc count

    sorted_wcs = dict(sorted(roster_wcs.items(), key=lambda item: item[1], reverse=True))
    print("Raw WC counts for your roster:")
    pprint(sorted_wcs, indent=4, sort_dicts=False)

    sorted_wcs_trimmed = {k: v for k, v in sorted_wcs.items() if v <= ace_threshold - tolerance}
    print(f"Assuming that this is the last day you can ace for the week (which is the best time to ace, I think), I recommend acing {next(iter(sorted_wcs_trimmed))}")

# Predicts the coefficients of the heuristic formula for more accurate roster rankings
# The measured criteria have to be manually filled in. The JSON structure can be accessed in api/update_json.py
def criteria_prediction(current_criteria, roster):
    return_dict = {}

    for title in roster:
        return_dict[title] = {}
        temp_measurements = np.zeros((len(current_criteria), 5))

        for day in current_criteria:
            temp_measurements[day][0] = current_criteria.get(day, {}).get(title, {}).get("score", 0)
            temp_measurements[day][1] = current_criteria.get(day, {}).get(title, {}).get("wc", 0)
            temp_measurements[day][2] = current_criteria.get(day, {}).get(title, {}).get("favorites", 0)
            temp_measurements[day][3] = current_criteria.get(day, {}).get(title, {}).get("dropped", 0)
            temp_measurements[day][4] = current_criteria.get(day, {}).get(title, {}).get("forum", 0)

        temp_observation = np.eye(5)
        temp_noise = np.eye(5) # CHANGE THIS TO ESTIMATED NOISE
        temp_covs = sample_covariance(current_criteria, 5)
        temp_transition = np.eye(5)
        
        temp_kalman = KalmanFilter(temp_transition, temp_observation, temp_noise, temp_covs, temp_measurements)
        if not temp_kalman.check():
                print("Check failed for criteria!")
                return None
        temp_preds = temp_kalman.make_predictions()
        return_dict[title]['score'] = temp_preds[0]
        return_dict[title]['wc'] = temp_preds[1]
        return_dict[title]['favorites'] = temp_preds[2]
        return_dict[title]['dropped'] = temp_preds[3]
        return_dict[title]['forum'] = temp_preds[4]

    print("Criteria distribution prediction success!")

    return return_dict

# Z-score predictions for your roster
def main_prediction(current_json, roster):
    current_metrics = {}

    return_dict = {}

    for day in current_json:
        current_metrics[day] = {}

        for title in current_json.get(day, {}):
            current_metrics[day][title] = dict(list(current_json.get(day, {}).get(title, {}).items())[:8])
    
    for title in roster:
        temp_measurements = np.zeros((len(current_metrics), 8))
        return_dict[title] = {}

        for day in current_metrics:
            temp_measurements[day][0] = current_metrics.get(day, {}).get(title, {}).get("score", 0)
            temp_measurements[day][1] = current_metrics.get(day, {}).get(title, {}).get("forum", 0)
            temp_measurements[day][2] = current_metrics.get(day, {}).get(title, {}).get("dropped", 0)
            temp_measurements[day][3] = current_metrics.get(day, {}).get(title, {}).get("dropped_dot", 0)
            temp_measurements[day][4] = current_metrics.get(day, {}).get(title, {}).get("wc", 0)
            temp_measurements[day][5] = current_metrics.get(day, {}).get(title, {}).get("wc_dot", 0)
            temp_measurements[day][6] = current_metrics.get(day, {}).get(title, {}).get("favorites", 0)
            temp_measurements[day][7] = current_metrics.get(day, {}).get(title, {}).get("favorites_dot", 0)

        temp_observation = np.eye(8)
        temp_noise = np.eye(8) # CHANGE THIS TO ESTIMATED NOISE
        temp_covs = sample_covariance(current_metrics, 8)
        temp_transition = np.array([[0.8, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0.8, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 1]])
        
        temp_kalman = KalmanFilter(temp_transition, temp_observation, temp_noise, temp_covs, temp_measurements)
        if not temp_kalman.check():
                print("Check failed for main predictions!")
                return None
        temp_preds = temp_kalman.make_predictions()
        return_dict[title]['score'] = temp_preds[0]
        return_dict[title]['forum'] = temp_preds[1]
        return_dict[title]['dropped'] = temp_preds[2]
        return_dict[title]['dropped_dot'] = temp_preds[3]
        return_dict[title]['wc'] = temp_preds[4]
        return_dict[title]['wc_dot'] = temp_preds[5]
        return_dict[title]['favorites'] = temp_preds[6]
        return_dict[title]['favorites_dot'] = temp_preds[7]

    return return_dict

# Updates roster
def roster_eval(main_dict, criteria_dict):
    final_scores = {}

    for title in main_dict:
        final_scores[title] = 0

        for metric in ['score', 'wc', 'favorites', 'dropped', 'forum']:
            final_scores[title] += main_dict.get(title, {}).get(metric, 0) * criteria_dict.get(title, {}).get(metric, 0)

    sorted_scores = dict(sorted(final_scores.items(), key=lambda item: item[1], reverse=True))
    print("Predicted metrics for your roster:")
    pprint(sorted_scores, indent=4, sort_dicts=False)

# Ties all the functions together
def pipeline(current_json, criteria_json, roster, ace_threshold, tolerance):
    acing_predictions(current_json, roster, ace_threshold, tolerance)
    criteria_dict = criteria_prediction(criteria_json, roster)
    main_dict = main_prediction(current_json, roster)
    roster_eval(main_dict, criteria_dict)

# Tests Claude-generated synthetic data
def claude_test():
    target_dir = Path("data/kalman_predictions")
    current_path = target_dir / "claude_current.json"
    criteria_path = target_dir / "claude_criteria.json"

    roster = [
    "Solo Leveling S2",
    "Frieren",
    "Jujutsu Kaisen S3",
    "Blue Lock S2",
    "Wind Breaker S2"
    ]

    current_json = {}
    criteria_json = {}

    try:
        with current_path.open("r", encoding="utf-8") as file:
            current_json = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file at {current_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file at {current_path} contains invalid JSON formatting.")

    try:
        with criteria_path.open("r", encoding="utf-8") as file:
            criteria_json = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file at {criteria_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file at {criteria_path} contains invalid JSON formatting.")

    current_json = {int(k): v for k, v in current_json.items()}
    criteria_json = {int(k): v for k, v in criteria_json.items()}

    pipeline(current_json, criteria_json, roster, ace_threshold=60_000, tolerance=1_000)

if __name__ == "__main__":
    claude_test()