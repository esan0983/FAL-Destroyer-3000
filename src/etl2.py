# Partially assisted by Claude

import requests
import json
import time
import pandas as pd
import os

from utils import base_url, get_anime, get_anime_name, get_anime_episodes, get_anime_statistics, get_prequel


# ---------------------------------------------------------------------------
# Per-ID extraction (replaces the page-based retrieve_data/extract_data logic)
# ---------------------------------------------------------------------------

COLUMNS = [
    'mal_id', 'title', 'source', 'episodes', 'synopsis', 'year',
    'season', 'producers', 'genres', 'studios', 'demographics',
    'themes', 'rating', 'sequel', 'favorites', 'score', 'wc',
    'dropped', 'forum', 'type'
]

def extract_single(id_num):
    """
    Pull every field needed for one MAL ID, mirroring the field set etl.py
    collected per page-row. Returns a dict matching COLUMNS, or None if the
    anime doesn't exist / isn't a usable row.
    """
    anime_json = get_anime(id_num)
    if anime_json is None:
        return None  # ID doesn't exist, caller treats this as a "miss"

    anime = anime_json.get('data', {})
    if not anime:
        return None

    # Only keep TV entries, same filter transform_data() applied before
    if anime.get('type') != "TV":
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
    anime_type = anime.get('type')

    producers = [p['name'] for p in anime.get('producers', [])]
    genres = [g['name'] for g in anime.get('genres', [])]
    studios = [s['name'] for s in anime.get('studios', [])]
    demographics = [d['name'] for d in anime.get('demographics', [])]
    themes = [t['name'] for t in anime.get('themes', [])]

    time.sleep(1.1)
    sequel = get_prequel(id_num)

    time.sleep(1.1)
    stat_json = get_anime_statistics(id_num)
    if stat_json is not None:
        wc = stat_json['data']['watching'] + stat_json['data']['completed']
        dropped = stat_json['data']['dropped']
    else:
        wc = None
        dropped = None

    time.sleep(1.1)
    eps_json = get_anime_episodes(id_num)
    forum = sum(
        ep.get('replies', 0)
        for ep in eps_json.get('data', [])
        if ep.get('mal_id', 0) <= 13
    )

    time.sleep(1.1)

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
        'type': anime_type,
    }
    return row


def load_data(df):
    print("Starting data loading...")
    folder_path = "data/raw"
    os.makedirs(folder_path, exist_ok=True)
    file_name = "anime_data.csv"
    full_path = os.path.join(folder_path, file_name)
    df.to_csv(full_path, index=False)
    print(f"File successfully saved to: {full_path}")
    print(df.info())
    return df


def run(start_id, df, max_id, max_consecutive_misses, save_every):
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
            result = extract_single(current_id)
        except ValueError as e:
            # Rate-limited or server error: back off and retry same ID
            print(f"Error on id {current_id}: {e}. Backing off 5s and retrying...")
            time.sleep(5)
            continue

        if result is None:
            consecutive_misses += 1
            print(f"id {current_id}: not found ({consecutive_misses}/{max_consecutive_misses} consecutive misses)")
        elif result == "SKIP":
            consecutive_misses = 0
            print(f"id {current_id}: exists but not TV type, skipping")
        else:
            consecutive_misses = 0
            pending_rows.append(result)
            print(f"id {current_id}: collected '{result['title']}'")

        if len(pending_rows) >= save_every:
            new_data = pd.DataFrame(pending_rows, columns=COLUMNS)
            df = pd.concat([df, new_data], ignore_index=True)
            df = load_data(df)
            pending_rows = []

        current_id += 1

    # Flush any remaining rows
    if pending_rows:
        new_data = pd.DataFrame(pending_rows, columns=COLUMNS)
        df = pd.concat([df, new_data], ignore_index=True)
        df = load_data(df)

    print("Finished! Confirmation below:")
    print(f"Last id attempted: {current_id}")
    print(f"Consecutive misses at stop: {consecutive_misses}")
    return df


if __name__ == "__main__":
    initial_data = pd.read_csv("data/raw/anime_data.csv")

    START_ID = 54804  # resume point after page-based crawl stalled at page 1000
    # Set MAX_ID if you want a hard ceiling; otherwise the miss-streak
    # threshold below will stop the crawl once it runs past real MAL IDs.
    MAX_ID = 70000
    MAX_CONSECUTIVE_MISSES = 1000
    SAVE_EVERY = 25

    run(
        start_id=START_ID,
        df=initial_data,
        max_id=MAX_ID,
        max_consecutive_misses=MAX_CONSECUTIVE_MISSES,
        save_every=SAVE_EVERY,
    )