# api/forum_collection.py
# this was made so that I don't have to redo etl2.py

import pandas as pd
import time

from api.api_utils import (
    get_anime_episodes
)

def get_forums(df):
    func_df = df

    for mal_id in df['mal_id']:
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

        print(f"13-episode forum reply count for MAL ID {mal_id}: {forum}")
        func_df.loc[func_df['mal_id'] == mal_id, 'forum'] = forum

    return func_df

if __name__ == "__main__":
    initial_data  = pd.read_parquet("data/processed/anime_data_1.parquet")
    final_data = get_forums(initial_data)
    final_data.to_parquet("data/processed/anime_data_1.parquet")
