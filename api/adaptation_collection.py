# DISCLAIMER: Mostly modified by Claude as I got lost and confused on code structure.

import os
import time
import pandas as pd

from api.api_utils import (
    get_manga, 
    get_anime_relations
)

target_media_types = {
    "Light Novel", "Manga", "Novel", "Doujinshi", "One-shot", "Manhwa", "Manhua"
}

INPUT_PATH = "data/processed/anime_data_1.parquet"
OUTPUT_PATH = "data/processed/adaptations.parquet"


def get_adaptation_entries(relations_json):
    """Return all adaptation entries (across all source types, e.g. 'manga') that
    match one of our target media types."""
    return [
        entry
        for rel in relations_json.get("data", [])
        if rel.get("relation") == "Adaptation"
        for entry in rel.get("entry", [])
        if entry.get("media_type") in target_media_types
    ]


def collect_adaptations(mal_ids):
    score_lists = {mt: [] for mt in target_media_types}
    member_lists = {mt: [] for mt in target_media_types}

    for i, mal_id in enumerate(mal_ids):
        print(f"Processing mal_id {mal_id}...")

        relations_json = None
        relations_success = False
        while not relations_success:
            try:
                time.sleep(0.34)
                relations_json = get_anime_relations(mal_id)
                print(f"Collected relations for {mal_id}!")
                relations_success = True
            except Exception as e:
                print(f"Failed to get relations for {mal_id}: {e}")
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    print(f"Rate limited on mal_id {mal_id}. Backing off 5s and retrying...")
                    time.sleep(5)
                else:
                    for mt in target_media_types:
                        score_lists[mt].append(None)
                        member_lists[mt].append(None)
                    relations_success = True
                    relations_json = None
                    break

        if relations_json is None:
            continue

        adaptations = get_adaptation_entries(relations_json)

        for mt in target_media_types:
            matching_entry = next(
                (entry for entry in adaptations
                 if entry.get("media_type", "").lower() == mt.lower()),
                None
            )

            if matching_entry is not None:
                adaptation_id = matching_entry.get("mal_id")
                request_success = False
                while not request_success:
                    try:
                        time.sleep(0.34)
                        manga_json = get_manga(adaptation_id)
                        if manga_json is None:
                            print(f"No manga data for adaptation {adaptation_id} (source {mal_id}).")
                            score_lists[mt].append(None)
                            member_lists[mt].append(None)
                            request_success = True
                            continue
                        print(f"Called source material for {mal_id}! Source ID is {adaptation_id}.")
                        score_lists[mt].append(manga_json.get("data", {}).get("score"))
                        member_lists[mt].append(manga_json.get("data", {}).get("members"))
                        request_success = True
                    except Exception as e:
                        if getattr(e, "response", None) is not None and e.response.status_code == 429:
                            print(f"Rate limited on adaptation id {adaptation_id}. Backing off 5s and retrying...")
                            time.sleep(5)
                        else:
                            print(f"Failed to get manga data for adaptation {adaptation_id}: {e}")
                            score_lists[mt].append(None)
                            member_lists[mt].append(None)
                            request_success = True
            else:
                score_lists[mt].append(None)
                member_lists[mt].append(None)

        assert all(len(v) == i + 1 for v in score_lists.values())
        assert all(len(v) == i + 1 for v in member_lists.values())

    return score_lists, member_lists


def build_dataframe(mal_ids, score_lists, member_lists):
    data = {"mal_id": mal_ids}

    for mt in target_media_types:
        col_key = mt.lower().replace(" ", "_").replace("-", "_")
        data[f"{col_key}_score"] = score_lists[mt]
        data[f"{col_key}_members"] = member_lists[mt]

    return pd.DataFrame(data)

def main():
    df_ids = pd.read_parquet(INPUT_PATH)
    mal_ids = df_ids["mal_id"].tolist()

    score_lists, member_lists = collect_adaptations(mal_ids)
    result_df = build_dataframe(mal_ids, score_lists, member_lists)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved adaptation data to: {OUTPUT_PATH}")
    print(result_df.info())


if __name__ == "__main__":
    main()