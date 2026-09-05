# api/etl.2py
# Main file that performs all of the data collection
# custom_bool allows you to switch between current anime and all previous anime
# Partially assisted by Claude


import requests
import json
import time
import pandas as pd
import os
import numpy as np
from pathlib import Path
import re

from api.api_utils import (
    get_anime, 
    get_manga, 
    get_anime_episodes, 
    get_anime_statistics, 
    get_prequel, 
    get_ids
)
from api.adaptation_collection import (
    collect_adaptations, 
    target_media_types 
)   








COLUMNS = [
    'mal_id', 'title', 'source', 'episodes', 'cohort', 'genres',  'demographics',
    'themes', 'rating', 'sequel', 'prequel_id', 'favorites', 'score', 'wc',
    'dropped', 'forum', 'doujinshi_score',
    'doujinshi_members', 'manhua_score', 'manhua_members',
    'novel_score', 'novel_members', 'manhwa_score',
    'manhwa_members', 'light_novel_score', 'light_novel_members',
    'manga_score', 'manga_members', 'one_shot_score',
    'one_shot_members', 'studios', 'producers'
]

def extract_single(id_num, custom_bool):
    anime_json = None
    get_anime_success = False
    time.sleep(0.34)
    while not get_anime_success:
        try:
            anime_json = get_anime(id_num)
            print(f"get_anime successful for {id_num}!")
            get_anime_success = True
        except Exception as e:
            print(f"get_anime failed for {id_num}: {e}")
            if getattr(e, "response", None) is not None and e.response.status_code == 429:
                print(f"Rate limited on ID {id_num}. Backing off 5s and retrying...")
                time.sleep(5)
            else:
                break
                
    if anime_json is None:
        return None  # ID doesn't exist, caller treats this as a "miss"

    anime = anime_json.get('data', {})
    if not anime:
        return None
        
    # Criteria from data cleaning:
    # 1. Must be a TV series
    # 2. Must not be currently airing
    # 3. Must have a score (justification in README)
    # 4. Must have a year
    # 5. Must have a season
    if (not custom_bool and
        (anime.get('type') != "TV" or
        anime.get('airing') or
        anime.get('score') is None or 
        anime.get('year') is None or
        anime.get('season') is None)
    ):
        return "SKIP"  # exists, but filtered out downstream — not a "miss"

    title = anime.get('title')
    source = anime.get('source')
    episodes = anime.get('episodes')
    year = anime.get('year')
    season = anime.get('season')
    rating = anime.get('rating')
    favorites = anime.get('favorites')
    score = anime.get('score')

    cohort = season + " " + str(year)

    genres = [g['name'] for g in anime.get('genres', [])]
    demographics = [d['name'] for d in anime.get('demographics', [])]
    themes = [t['name'] for t in anime.get('themes', [])]
    studios = [d['name'] for d in anime.get('studios', [])]
    producers = [t['name'] for t in anime.get('producers', [])]


    # GET PREQUEL DATA
    # NOTE: PAY ATTENTION TO PREQUEL TYPE IN DATA CLEANING NOTEBOOK. WE ARE Z-SCORING BY COHORT, SO IT'S IMPORTANT TO
    # REMOVE NON-TV PREQUELS
    sequel = False
    prequel_id = None
    get_prequel_success = False
    time.sleep(0.34)
    while not get_prequel_success:
        try:   
            sequel, prequel_id = get_prequel(id_num)
            print(f"get_prequel successful for {id_num}! Prequel ID is {prequel_id}")
            get_prequel_success = True
        except Exception as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 429:
                print(f"Rate limited on ID {id_num}. Backing off 5s and retrying...")
                time.sleep(5)
            else:
                break
         

    # GET ADAPTATION INFORMATION FROM ADAPTATION_COLLECTION.PY
    # COLLECT ADAPTATIONS ACCEPTS A LIST OF IDS, SO WE TAKE ID_NUM TO BE A ONE ELEMENT ARRAY
    temp_list = [id_num]
    score_list, member_list = collect_adaptations(temp_list)

    # TURN INTO KEY AND VALUE
    adaptation_data = {}
    for mt in target_media_types:
        col_key = mt.lower().replace(" ", "_").replace("-", "_")

        if score_list[mt][0] is not None:
            adaptation_data[f"{col_key}_score"] = score_list[mt][0]
        else:
            adaptation_data[f"{col_key}_score"] = None

        if member_list[mt][0] is not None:
            adaptation_data[f"{col_key}_members"] = member_list[mt][0]
        else:
            adaptation_data[f"{col_key}_members"] = None

    time.sleep(0.34)

    # COLLECT ANIME STATISTICS
    get_statistics_success = False
    stat_json = None
    while not get_statistics_success:
        try:
            stat_json = get_anime_statistics(id_num)
            print(f"get_anime_statistics successful for {id_num}!")
            get_statistics_success = True
        except Exception as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 429:
                print(f"Rate limited on ID {id_num}. Backing off 5s and retrying...")
                time.sleep(5)
            else:
                break
            
    if stat_json is not None:
        wc = stat_json.get('data').get('watching') + stat_json.get('data').get('completed')
        dropped = stat_json.get('data').get('dropped')
    else:
        wc = None
        dropped = None

    # WILL FIX IN THE FUTURE, BUGGED FOR NOW
    get_episodes_success = False
    while not get_episodes_success:
        try:
            eps_json = get_anime_episodes(id_num)
            print(f"get_anime_episodes successful for {id_num}!")
            get_episodes_success = True
        except Exception as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 429:
                print(f"Rate limited on ID {id_num}. Backing off 5s and retrying...")
                time.sleep(5)
            else:
                break

    forum = sum(
        ep.get('replies', 0)
        for ep in eps_json.get('data', [])
        if ep.get('mal_id', 0) <= 13
    )

    # IMAGE EXTRACTION
    # img_path = "data/images"
    # image_url = anime.get('images').get('jpg').get('large_image_url')
    # if image_url is not None:
    #     file_name = str(id_num) + ".jpg"
    #     download_image(image_url, img_path, file_name)
    #     thumbnail = True
    # else:
    #     thumbnail = False

    row = {
        'mal_id': id_num,
        'title': title,
        'source': source,
        'episodes': episodes,
        'cohort': cohort,
        'genres': genres,
        'demographics': demographics,
        'themes': themes,
        'rating': rating,
        'sequel': sequel,
        'prequel_id' : prequel_id,
        'favorites': favorites,
        'score': score,
        'wc': wc,
        'dropped': dropped,
        'forum': forum,
        'studios': studios,
        'producers': producers
    }

    # CONCATENATE OTHER DICTS
    final_row = row | adaptation_data 

    return final_row


def load_data(df, custom_bool):
    print("Starting data loading...")
    folder_path = "data/raw"
    os.makedirs(folder_path, exist_ok=True)
    file_name = "current_data.csv" if custom_bool else "anime_data.csv"
    full_path = os.path.join(folder_path, file_name)
    df.to_csv(full_path, index=False)
    print(f"File successfully saved to: {full_path}")
    print(df.info())
    return df


def run(start_id, df, max_id, max_consecutive_misses, save_every, custom_bool):
    """
    Walk MAL IDs upward starting at start_id, one anime at a time, instead of
    the old page-based crawl (which caps out at page 1000 for this API tier).

    Stops when:
      - `max_id` is reached (if provided), or
      - we hit `max_consecutive_misses` IDs in a row that don't exist,
        which is our signal we've run past the end of currently-registered
        MAL entries.
    """
    print("ETL Pipeline (ID-based) Currently Running!")
    current_id = start_id
    consecutive_misses = 0
    pending_rows = []

    while consecutive_misses < max_consecutive_misses:
        if max_id is not None and current_id > max_id:
            print(f"Reached max_id={max_id}, stopping.")
            break
        try:
            result = extract_single(current_id, custom_bool)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limited on id {current_id}. Backing off 5s and retrying...")
                time.sleep(5)
                continue 
            else:
                print(f"Unhandled HTTP error on id {current_id}: {e}. Skipping.")

        if result is None:
            consecutive_misses += 1
            print(f"id {current_id}: not found ({consecutive_misses}/{max_consecutive_misses} consecutive misses)")
        elif result == "SKIP":
            consecutive_misses = 0
            print(f"id {current_id}: exists but breaks criteria, skipping")
        else:
            consecutive_misses = 0
            pending_rows.append(result)
            print(f"id {current_id}: collected '{result['title']}'")

        if len(pending_rows) >= save_every:
            new_data = pd.DataFrame(pending_rows, columns=COLUMNS)
            df = pd.concat([df, new_data], ignore_index=True)
            df = load_data(df, custom_bool)
            pending_rows = []

        current_id += 1

    # Flush any remaining rows
    if pending_rows:
        new_data = pd.DataFrame(pending_rows, columns=COLUMNS)
        df = pd.concat([df, new_data], ignore_index=True)
        df = load_data(df, custom_bool)

    print("Finished! Confirmation below:")
    print(f"Last id attempted: {current_id}")
    print(f"Consecutive misses at stop: {consecutive_misses}")
    return df

def run_custom(df, mal_ids, save_every, custom_bool=True):
    """
    Collect Fall 2026 Data
    """
    print("ETL Pipeline (ID-based) Currently Running!")
    pending_rows = []

    for current_id in mal_ids:
        try:
            result = extract_single(current_id, custom_bool)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limited on id {current_id}. Backing off 5s and retrying...")
                time.sleep(5)
                continue 
            else:
                print(f"Unhandled HTTP error on id {current_id}: {e}. Skipping.")

        pending_rows.append(result)

        if len(pending_rows) >= save_every:
            new_data = pd.DataFrame(pending_rows, columns=COLUMNS)
            df = pd.concat([df, new_data], ignore_index=True)
            df = load_data(df, custom_bool)
            pending_rows = []

    # Flush any remaining rows
    if pending_rows:
        new_data = pd.DataFrame(pending_rows, columns=COLUMNS)
        df = pd.concat([df, new_data], ignore_index=True)
        df = load_data(df, custom_bool)

    print("Finished!")
    return df

def run_custom2(df, custom_ids, custom_bool=False):
    print("ETL Pipeline (Custom IDS) Currently Running!")
    pending_rows = []

    for current_id in custom_ids:
        try:
            result = extract_single(current_id, custom_bool)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limited on id {current_id}. Backing off 5s and retrying...")
                time.sleep(5)
                continue 
            else:
                print(f"Unhandled HTTP error on id {current_id}: {e}. Skipping.")

        pending_rows.append(result)

    # Flush any remaining rows
    if pending_rows:
        new_data = pd.DataFrame(pending_rows, columns=COLUMNS)
        df = pd.concat([df, new_data], ignore_index=True)
        df = load_data(df, custom_bool)

    print("Finished!")
    return df


if __name__ == "__main__":
    custom_bool = False # CHANGE THIS FOR EITHER STANDARD COLLECTION OR FALL 2026 COLLECTION
    initial_data = pd.read_csv("data/raw/current_data.csv") if custom_bool else pd.read_csv("data/raw/anime_data.csv")

    START_ID = 62914  # resume point
    # Set MAX_ID if you want a hard ceiling; otherwise the miss-streak
    # threshold below will stop the crawl once it runs past real MAL IDs.
    MAX_ID = 66000
    MAX_CONSECUTIVE_MISSES = 2500
    SAVE_EVERY = 25

    custom_ids = [30091, 29836, 29974, 29976, 29854, 29865, 30123, 30127, 29758, 30015, 30016, 30144, 30028, 30156, 30030, 30039, 29785, 29786, 29787, 30173, 29803, 30187, 29941, 30205]

    # FOR STANDARD COLLECTION
    # run(
    #     start_id=START_ID,
    #     df=initial_data,
    #     max_id=MAX_ID,
    #     max_consecutive_misses=MAX_CONSECUTIVE_MISSES,
    #     save_every=SAVE_EVERY,
    #     custom_bool=custom_bool
    # )

    # FOR FALL 2026 COLLECTION
    mal_ids = get_ids()
    run_custom(
        df=initial_data,
        mal_ids=mal_ids,
        save_every=SAVE_EVERY,
        custom_bool=True
    )

    # FOR CUSTOM ID COLLECTION
    # run_custom2(
    #     df=initial_data,
    #     custom_ids=custom_ids,
    #     custom_bool=False
    # )