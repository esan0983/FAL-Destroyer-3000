# Partially assisted by Claude
# Planning to integrate some of the data cleaning protocols, image extraction, prequel data, and adaptation collection for a final data pass

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
from utils import download_image

pattern = r"\s*[\(\[].*?[\)\]]\s*$"

# ---------------------------------------------------------------------------
# Source material initialization (takes mean and stdev of source material score and log1p'd member count)
# ---------------------------------------------------------------------------
init_dir = Path("data/stats")
init_score = init_dir / "scores.json"
init_members = init_dir / "members.json"

try:
    with open(init_score, "r", encoding="utf-8") as file:
        score_data = json.load(file)
except FileNotFoundError:
    print(f"Error: The file at {init_score} was not found.")
except json.JSONDecodeError:
    print("Error: The file contains invalid JSON.")

try:
    with open(init_members, "r", encoding="utf-8") as file:
        member_data  = json.load(file)
except FileNotFoundError:
    print(f"Error: The file at {init_members} was not found.")
except json.JSONDecodeError:
    print("Error: The file contains invalid JSON.")

score_means = {}
score_stdevs = {}
member_means = {}
member_stdevs = {}

for mt in target_media_types:
    score_means[mt] = np.mean(score_data[mt])
    score_stdevs[mt] = np.std(score_data[mt])
    member_means[mt] = np.mean(np.log1p(member_data[mt]))
    member_stdevs[mt] = np.std(np.log1p(member_data[mt]))


# ---------------------------------------------------------------------------
# Per-ID extraction (replaces the page-based retrieve_data/extract_data logic)
# ---------------------------------------------------------------------------

COLUMNS = [
    'mal_id', 'title', 'source', 'episodes', 'synopsis', 'year',
    'season', 'producers', 'genres', 'studios', 'demographics',
    'themes', 'rating', 'sequel', 'favorites', 'score', 'wc',
    'dropped', 'forum', 'members', 'thumbnail', 'doujinshi_score_z',
    'doujinshi_members_z', 'manhua_score_z', 'manhua_members_z',
    'novel_score_z', 'novel_members_z', 'manhwa_score_z',
    'manhwa_members_z', 'light_novel_score_z', 'light_novel_members_z',
    'manga_score_z', 'manga_members_z', 'one_shot_score_z',
    'one_shot_members_z', 'prequel_score', 'prequel_members', 'prequel_type'
]

def extract_single(id_num, custom_bool):
    """
    Pull every field needed for one MAL ID, mirroring the field set etl.py
    collected per page-row. Returns a dict matching COLUMNS, or None if the
    anime doesn't exist / isn't a usable row.
    """
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
    synopsis = anime.get('synopsis')
    year = anime.get('year')
    season = anime.get('season')
    rating = anime.get('rating')
    favorites = anime.get('favorites')
    score = anime.get('score')
    members = anime.get('members') # FOR PREQUEL Z-SCORING PURPOSES

    if synopsis is not None:
        synopsis = re.sub(pattern, "", synopsis)

    producers = [p['name'] for p in anime.get('producers', [])]
    genres = [g['name'] for g in anime.get('genres', [])]
    studios = [s['name'] for s in anime.get('studios', [])]
    demographics = [d['name'] for d in anime.get('demographics', [])]
    themes = [t['name'] for t in anime.get('themes', [])]


    # GET PREQUEL DATA
    # NOTE: PAY ATTENTION TO PREQUEL TYPE IN DATA CLEANING NOTEBOOK. WE ARE Z-SCORING BY COHORT, SO IT'S IMPORTANT TO
    # REMOVE NON-TV PREQUELS
    sequel = None
    prequel_id = None
    get_prequel_success = False
    time.sleep(0.34)
    while not get_prequel_success:
        try:   
            sequel, prequel_id = get_prequel(id_num)
            print(f"get_prequel successful for {id_num}!")
            get_prequel_success = True
        except Exception as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 429:
                print(f"Rate limited on ID {id_num}. Backing off 5s and retrying...")
                time.sleep(5)
            else:
                break

    time.sleep(0.34)          
    
    prequel_data = {}
    if prequel_id is None:
        prequel_data['prequel_score'] = None
        prequel_data['prequel_members'] = None
        prequel_data['prequel_type'] = None
    else:
        get_anime_prequel_success = False
        prequel_json = {}
        while not get_anime_prequel_success:
            try:
                prequel_json = get_anime(prequel_id)
                print(f"get_anime for prequel successful for {prequel_id}!")
                get_anime_prequel_success = True
            except Exception as e:
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    print(f"Rate limited on prequel ID {prequel_id}. Backing off 5s and retrying...")
                    time.sleep(5)
                else:
                    break

        prequel = prequel_json.get('data', {})

        prequel_data['prequel_score'] = prequel.get('score')
        prequel_data['prequel_members'] = prequel.get('members')
        prequel_data['prequel_type'] = prequel.get('type')

    # GET ADAPTATION INFORMATION FROM ADAPTATION_COLLECTION.PY
    # COLLECT ADAPTATIONS ACCEPTS A LIST OF IDS, SO WE TAKE ID_NUM TO BE A ONE ELEMENT ARRAY
    temp_list = [id_num]
    score_list, member_list = collect_adaptations(temp_list)

    # TURN INTO KEY AND VALUE
    adaptation_data = {}
    for mt in target_media_types:
        col_key = mt.lower().replace(" ", "_").replace("-", "_")

        if score_list[mt][0] is not None:
            adaptation_data[f"{col_key}_score_z"] = (score_list[mt][0] - score_means[mt]) / score_stdevs[mt]
        else:
            adaptation_data[f"{col_key}_score_z"] = None

        if member_list[mt][0] is not None:
            adaptation_data[f"{col_key}_members_z"] = (np.log1p(member_list[mt][0]) - member_means[mt]) / member_stdevs[mt]
        else:
            adaptation_data[f"{col_key}_members_z"] = None

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
    # get_episodes_success = False
    # while not get_episodes_success:
    #     try:
    #         eps_json = get_anime_episodes(id_num)
    #         print(f"get_anime_episodes successful for {id_num}!")
    #         get_episodes_success = True
    #     except Exception as e:
    #         if getattr(e, "response", None) is not None and e.response.status_code == 429:
    #             print(f"Rate limited on ID {id_num}. Backing off 5s and retrying...")
    #             time.sleep(5)
    #         else:
    #             break

    # if eps_json is not None:
    #     forum = sum(
    #         ep.get('replies', 0)
    #         for ep in eps_json.get('data', [])
    #         if ep.get('mal_id', 0) <= 13
    #     )
    # else:
    #     forum = 0

    forum = 0

    # IMAGE EXTRACTION
    img_path = "data/images"
    image_url = anime.get('images').get('jpg').get('large_image_url')
    if image_url is not None:
        file_name = str(id_num) + ".jpg"
        download_image(image_url, img_path, file_name)
        thumbnail = True
    else:
        thumbnail = False

    row = {
        'mal_id': id_num,
        'title': title,
        'source': source,
        'episodes': episodes,
        'synopsis': synopsis,
        'year': year,
        'season': season,
        'producers': producers,
        'genres': genres,
        'studios': studios,
        'demographics': demographics,
        'themes': themes,
        'rating': rating,
        'sequel': sequel,
        'favorites': favorites,
        'score': score,
        'wc': wc,
        'dropped': dropped,
        'forum': forum,
        'members': members,
        'thumbnail': thumbnail
    }

    # CONCATENATE OTHER DICTS
    final_row = row | adaptation_data | prequel_data

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

def run_custom(df, mal_ids, save_every, custom_bool):
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


if __name__ == "__main__":
    custom_bool = False # CHANGE THIS FOR EITHER STANDARD COLLECTION OR FALL 2026 COLLECTION
    initial_data = pd.read_csv("data/raw/current_data.csv") if custom_bool else pd.read_csv("data/raw/anime_data.csv")

    START_ID = 1  # resume point
    # Set MAX_ID if you want a hard ceiling; otherwise the miss-streak
    # threshold below will stop the crawl once it runs past real MAL IDs.
    MAX_ID = 75000
    MAX_CONSECUTIVE_MISSES = 50
    SAVE_EVERY = 25

    # FOR STANDARD COLLECTION
    run(
        start_id=START_ID,
        df=initial_data,
        max_id=MAX_ID,
        max_consecutive_misses=MAX_CONSECUTIVE_MISSES,
        save_every=SAVE_EVERY,
        custom_bool=custom_bool
    )

    # FOR FALL 2026 COLLECTION
    # mal_ids = get_ids()
    # run_custom(
    #     df=initial_data,
    #     mal_ids=mal_ids,
    #     save_every=SAVE_EVERY,
    #     custom_bool=custom_bool
    # )