# After practicing, we will imitate an ETL pipeline.

import requests
import json
import random
import time

base_url = "https://api.tenrai.org/v1"

def get_anime(id): # for JSON inspection
    url = f"{base_url}/anime/{id}"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        print("Data retrieved!")
        return anime_data
    else:
        print(f"Failed to retrieve data {response.status_code}")
        return None

def get_anime_name(id):
    url = f"{base_url}/anime/{id}"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        anime_name = anime_data['data']['title']
        print("Data retrieved!")
        return anime_name
    else:
        print(f"Failed to retrieve data {response.status_code}")
        return None

def get_anime_episodes(id):
    url = f"{base_url}/anime/{id}/episodes"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        print("Data retrieved!")
        return anime_data
    else:
        print(f"Failed to retrieve data {response.status_code}")
        return None

def get_anime_statistics(id):
    url = f"{base_url}/anime/{id}/statistics"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        return anime_data
    else:
        print(f"Failed to retrieve data {response.status_code}")
        return None

# anime_titles = []
# for i in range(20):
#     num = random.randint(1, 63100)
#     print(f"id: {num}")
#     anime_name = get_anime_name(num)
#     if anime_name != None:
#         anime_titles.append(get_anime_name(num))
#     time.sleep(1)

# print(anime_titles)

# print(json.dumps(get_anime(37987), indent=4))
# print(json.dumps(get_anime_episodes(33352), indent=4))
# print(json.dumps(get_anime_statistics(33352), indent=4))

# How to get data:
# Thumbnail: will deal with later
# Title: refer to get_anime_name
# Source: anime_data['data']['source']
# Episode Count: anime_data['data']['episodes']
# Synopsis: anime_data['data']['synopsis']
# Season & year: anime_data['data']['year'] and same for season
# Producers: access producers via anime_data['data']['producers'] and get all the names
# Studios: same as above but with anime_data['data']['studios']
# Genres: same as above but with anime_data['data']['genre']
# Age rating: anime_data['data']['rating']
# Is it a sequel: access via anime_data["relations"] and check if there exists a value "Prequel" for the key "relation"
# Streaming platform availability: rate limit is 30 per minute. Might have to do some task scheduling. Will need testing
# Demographics and themes maybe?
# Score: anime_data['data']['score']
# Watching + Completed: anime_stats['data']['watching'] + same thing but for completed
# 13-episode forum messages: access via anime_episodes['data'] and sum ['replies'] for ['mal_id'] <= 13
# Dropped same as completed and watching
# Favorites: anime_data['data']['favorites']

def get_anime_page():
    url = f"{base_url}/anime"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        print("Data retrieved!")
        return anime_data
    else:
        print(f"Failed to retrieve data {response.status_code}")
        return None

def get_prequel(id):
    url = f"{base_url}/anime/{id}/relations"
    response = requests.get(url)

    if response.status_code == 200:
        relation_data = response.json()
        relations = [r['relation'] for r in relation_data.get('data', [])]
        return (True if "Prequel" in relations else False)
    else:
        return False


print(json.dumps(get_anime_page(), indent=4))

requests = 0
max_pages = 10000
current_page = 1
page_array = []

# Data extraction
def extract_data(page_number):
    url = f"{base_url}/anime"
    params = {"page": page_number}
    response = requests.get(url)

    if response.status_code == 200:
        page_data = response.json()
        print(f"Data retreived for page {page_number}!")

        # Get ids to call other endpoints
        id_array = [anime['id'] for anime in page_data.get("data", [])]

        # Get title
        title_array = [anime['name'] for anime in page_data.get("data", [])]

        # Get sources
        source_array = [anime['source'] for anime in page_data.get("data", [])]

        # Get episode counts
        episode_array = [anime['episodes'] for anime in page_data.get("data", [])]

        # Get synopsis
        synopsis_array = [anime['synopsis'] for anime in page_data.get("data", [])]

        # Get year & season
        year_array = [anime['year'] for anime in page_data.get("data", [])]
        season_array = [anime['season'] for anime in page_data.get("data", [])]

        # Get producers, genres, studios, demographics, themes
        producers_array = []
        genres_array = []
        studios_array = []
        dem_array = []
        themes_array = []
        for anime in page_data.get("data", []):
            producers = [p['name'] for p in anime.get('producers', [])]
            genres = [g['name'] for g in anime.get('genres', [])]
            studios = [s['name'] for s in anime.get('studios', [])]
            demographics = [d['name'] for d in anime.get('demographics', [])]
            themes = [t['name'] for t in anime.get('themes', [])]

            producers_array.append(producers)
            genres_array.append(genres)
            studios_array.append(themes)

        # Get age rating
        rating_array = [anime['rating'] for anime in page_data.get("data", [])]

        # Get bool if it's a sequel
        sequel_array = []
        for anime in page_data.get('data', []):
            id_num = anime['mal_id']
            sequel_array.append(get_prequel(id_num))
            time.sleep(1)

        # Get favorites
        favorites_array = [anime['favorites'] for anime in page_data.get("data", [])]

        # Get scores
        scores_array = [anime['score'] for anime in page_data.get("data", [])]

        # Get metrics from "statistics" call
        wc_array = []
        dropped_array = []
        for anime in page_data.get('data', []):
            id_num = anime['mal_id']
            stat_json = get_anime_statistics(id_num)

            wc_array.append(stat_json['data']['watching'] + stat_json['data']['completed'])
            dropped_array.append(stat_json['data']['dropped'])

            time.sleep(1)

        # Get metrics from "episodes" call
        forum_array = []
        for anime in page_data.get('data', []):
            id_num = anime['mal_id']
            eps_json = get_anime_episodes(id_num)
            requests += 1

            total_replies = sum(
                ep.get('replies', 0)
                for ep in eps_json.get('data', [])
                if ep.get('mal_id', 0) <= 13
            )

            forum_array.append(total_replies)
            time.sleep(1)

        # AniList GraphQL API call (might not be completely possible due to potential matching failures)

        # Check if there's a next page
        has_next = page_data.get("pagination", {}).get("has_next_page", False)

        # Final checks before proceeding
        arrays = [id_array, title_array, source_array, episode_array, synopsis_array, year_array, season_array, producers_array,
                  genres_array, studios_array, dem_array, themes_array, rating_array, sequel_array, favorites_array, scores_array,
                  wc_array, dropped_array, forum_array]

        all_same_length = all(len(lst) == len(arrays[0]) for lst in arrays)

        return (arrays if all_same_length else []), has_next
    else:
        return [], False



        