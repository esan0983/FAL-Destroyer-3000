# api/stats_collection.py
# Retrieves stats for every single entry under the MAL category "manga"
# Used for z-scoring purposes

from pathlib import Path
from utils import base_url, get_manga
import time
import json
import requests

def get_stats(id):
    manga_json = get_manga(id)
    if manga_json is None:
        return None, None, None
    
    manga = manga_json.get('data', {})
    if manga is None:
        return None, None, None

    return manga['score'], manga['members'], manga['type']

def run(starting_id, max_consecutive_misses, save_every, score_data, member_data):
    consecutive_misses = 0
    output_dir = Path("data/stats")
    score_path = output_dir / "scores.json"
    member_path = output_dir / "members.json"


    # CHANGE INITIALIZATION METHOD IF YOU EXPERIENCE A CRASH
    scores = score_data
    
    members = member_data

    current_id = starting_id
    
    while consecutive_misses <= max_consecutive_misses:
        score = None
        member_count = None
        media_type = None
        try:
            score, member_count, media_type = get_stats(current_id)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limited on id {current_id}. Backing off 5s and retrying...")
                time.sleep(5)
                continue 
            else:
                print(f"Unhandled HTTP error on id {current_id}: {e}. Skipping.")

        if score is None and member_count is None and media_type is None:
            print(f"Miss for ID {current_id}! ({consecutive_misses}/{max_consecutive_misses} consecutive misses)")
            consecutive_misses += 1
        elif media_type is None:
            print(f"No media type for {current_id}!")
            consecutive_misses = 0
        else:
            if score is not None:
                scores.setdefault(media_type, []).append(score)
            if member_count is not None:
                members.setdefault(media_type, []).append(member_count)
            consecutive_misses = 0
            print(f"Hit for ID {current_id}!")

        if current_id % save_every == 0:
            with open(score_path, "w") as file:
                json.dump(scores, file, indent=4)
            with open(member_path, "w") as file2:
                json.dump(members, file2, indent=4)

        current_id += 1

    with open(score_path, "w") as file:
        json.dump(scores, file, indent=4)
    with open(member_path, "w") as file2:
        json.dump(members, file2, indent=4)

    print("Finished! Confirmation below:")
    print(f"Last id attempted: {current_id}")
    print(f"Consecutive misses at stop: {consecutive_misses}")

if __name__ == "__main__":
    START_ID = 171501
    MAX_CONSECUTIVE_MISSES = 2000
    SAVE_EVERY = 500
    init_dir = Path("data/stats")
    init_score = init_dir / "scores.json"
    init_members = init_dir / "members.json"

    try:
        with open(init_score, "r", encoding="utf-8") as file:
            SCORE_DATA = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file at {init_score} was not found.")
    except json.JSONDecodeError:
        print("Error: The file contains invalid JSON.")

    try:
        with open(init_members, "r", encoding="utf-8") as file:
            MEMBER_DATA  = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file at {init_members} was not found.")
    except json.JSONDecodeError:
        print("Error: The file contains invalid JSON.") 


    run(
        starting_id=START_ID,
        max_consecutive_misses=MAX_CONSECUTIVE_MISSES,
        save_every=SAVE_EVERY,
        score_data=SCORE_DATA,
        member_data=MEMBER_DATA
    )