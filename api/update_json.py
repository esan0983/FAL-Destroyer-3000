# api/update_json.py
# Used to initialize and update stats of current anime

from pathlib import Path
from pprint import pprint
import json
import pandas as pd
import time
import numpy as np

from api.api_utils import (
    get_anime,
    get_anime_statistics,
    get_anime_episodes
)

def create_json(df):
    titles = df['title']

    current_stats = {
        0 : {
            
        }
    }

    for title in titles:
        current_stats.get(0, {})[title] = {}

    return current_stats

def load_json():
    target_dir = Path("../data/processed")
    file_path = target_dir / "current_stats.json"
    try:
        with file_path.open("r", encoding="utf-8") as file:
            current_stats = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} contains invalid JSON formatting.")

    return current_stats

def update_json(df, stats_json, day_num):
    current_stats = stats_json
    ids = df['mal_id']

    current_stats[day_num] = {}
    day = current_stats.get(day_num)
    prev_day = current_stats.get(day_num - 1, {})

    score_list = []
    wc_list = []
    favorites_list = []
    dropped_list = []
    forum_list = []

    for mal_id in ids:
        get_anime_success = False
        anime_json = None
        time.sleep(0.34)
        while not get_anime_success:
            try:
                anime_json = get_anime(mal_id)
                print(f"get_anime successful for {mal_id}!")
                get_anime_success = True
            except Exception as e:
                print(f"get_anime failed for {mal_id}: {e}")
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    print(f"Rate limited on ID {mal_id}. Backing off 5s and retrying...")
                    time.sleep(5)
                else:
                    break

        anime = anime_json.get('data', {})
        favorites = anime.get('favorites', 0)
        score = anime.get('score', None)
        score = np.nan if score is None else score

        get_statistics_success = False
        stat_json = None
        time.sleep(0.34)
        while not get_statistics_success:
            try:
                stat_json = get_anime_statistics(mal_id)
                print(f"get_anime_statistics successful for {mal_id}!")
                get_statistics_success = True
            except Exception as e:
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    print(f"Rate limited on ID {mal_id}. Backing off 5s and retrying...")
                    time.sleep(5)
                else:
                    break

        wc = stat_json.get('data').get('watching', 0) + stat_json.get('data').get('completed', 0)
        dropped = stat_json.get('data').get('dropped', 0)

        get_episodes_success = False
        time.sleep(0.34)
        while not get_episodes_success:
            try:
                eps_json = get_anime_episodes(mal_id)
                print(f"get_anime_episodes successful for {mal_id}!")
                get_episodes_success = True
            except Exception as e:
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    print(f"Rate limited on ID {mal_id}. Backing off 5s and retrying...")
                    time.sleep(5)
                else:
                    break

        forum = sum(
            ep.get('replies', 0)
            for ep in eps_json.get('data', [])
            if ep.get('mal_id', 0) <= 13
        )

        day[df.loc[df['mal_id'] == mal_id, 'title'].item()] = {
            "score" : score,
            "forum": forum, 
            "dropped": dropped, 
            "dropped_dot": 0,
            "wc": wc, 
            "wc_dot": 0,
            "favorites": favorites, 
            "favorites_dot": 0,
            "wc_raw": 0,
            "wc_raw_dot": 0
        }

        score_list.append(score)
        wc_list.append(wc)
        favorites_list.append(favorites)
        dropped_list.append(dropped)
        forum_list.append(forum)


    score_mean = np.nanmean(score_list)
    score_std = np.nanstd(score_list)

    wc_list = np.log1p(wc_list)
    wc_mean = np.nanmean(wc_list)
    wc_std = np.nanstd(wc_list)

    favorites_list = np.log1p(favorites_list)
    favorites_mean = np.nanmean(favorites_list)
    favorites_std = np.nanstd(favorites_list)

    dropped_list = np.log1p(dropped_list)
    dropped_mean = np.nanmean(dropped_list)
    dropped_std = np.nanstd(dropped_list)

    forum_list = np.log1p(forum_list)
    forum_mean = np.nanmean(forum_list)
    forum_std = np.nanstd(forum_list)

    for mal_id in ids:
        temp_score = day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('score')
        temp_wc = day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('wc')
        temp_favorites = day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('favorites')
        temp_dropped = day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('dropped')
        temp_forum = day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('forum')

        prev_wc = prev_day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('wc')
        prev_favorites = prev_day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('favorites')
        prev_dropped = prev_day.get(df.loc[df['mal_id'] == mal_id, 'title'].item()).get('dropped')

        day[df.loc[df['mal_id'] == mal_id, 'title'].item()] = {
            "score" : (temp_score - score_mean) / score_std,
            "forum": (temp_forum - forum_mean) / forum_std, 
            "dropped": (temp_dropped - dropped_mean) / dropped_std, 
            "dropped_dot": ((temp_dropped - dropped_mean) / dropped_std) - prev_dropped,
            "wc": (temp_wc - wc_mean) / wc_std, 
            "wc_dot": ((temp_wc - wc_mean) / wc_std) - prev_wc,
            "favorites": (temp_favorites - favorites_mean) / favorites_std, 
            "favorites_dot": ((temp_favorites - favorites_mean) / favorites_std) - prev_favorites
        }

    return current_stats

if __name__ == "__main__":
    day_num = 1
    df = pd.read_parquet("data/processed/current_data_1.parquet")
    current_json = create_json(df) # if you have not made a json yet
    current_json  = update_json(df, current_json, day_num)
    pprint(current_json, indent=4)