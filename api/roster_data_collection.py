# api/update_json.py
# Used to initialize and update stats of current anime

import pandas as pd
import time
import numpy as np

from api.api_utils import (
    get_anime,
    get_anime_statistics,
    get_anime_episodes,
    get_unique_users
)

def get_results(df: pd.DataFrame) -> None:
    ids = df['mal_id']

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
        df.loc[df['mal_id'] == mal_id, 'favorites'] = anime.get('favorites', 0)
        df.loc[df['mal_id'] == mal_id, 'score'] = anime.get('score', np.nan)

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

        df.loc[df['mal_id'] == mal_id, 'wc'] = stat_json.get('data').get('watching', 0) + stat_json.get('data').get('completed', 0)
        df.loc[df['mal_id'] == mal_id, 'dropped'] = stat_json.get('data').get('dropped', 0)

        get_episodes_success = False
        episodes_json = {}
        time.sleep(0.34)
        while not get_episodes_success:
            try:
                episodes_json = get_anime_episodes(mal_id)
                print(f"get_anime successful for {mal_id}!")
                get_episodes_success = True
            except Exception as e:
                print(f"get_anime failed for {mal_id}: {e}")
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    print(f"Rate limited on ID {mal_id}. Backing off 5s and retrying...")
                    time.sleep(5)
                else:
                    break

        ep_discussion_nums = df.loc[df['mal_id'], 'eps']
        ep_discussion_urls = [ep.get('forum_url', "")
                              for ep in episodes_json.get('data', [])
                              if ep.get('mal_id') in ep_discussion_nums]
        df['forum'] = sum(get_unique_users(url)
                          for url in ep_discussion_urls)

    print(df.head())

if __name__ == "__main__":
    # df should be manually edited in excel/sheets. It should have the following columns: mal_id, title, and eps
    # "eps" column contains the list of episode numbers that needs to be reviewed for forum scraping
    INITIAL_DF = pd.read_csv("data/raw/roster_data.csv")
    update_json = get_results(INITIAL_DF)
