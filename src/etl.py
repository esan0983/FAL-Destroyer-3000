import requests
import json
import random
import time
import pandas as pd
import os

base_url = "https://api.tenrai.org/v1"

def get_anime(id):
    url = f"{base_url}/anime/{id}"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        # print("Data retrieved!")
        return anime_data
    else:
        raise ValueError(f"Failed to retrieve data {response.status_code}")

def get_anime_name(id):
    url = f"{base_url}/anime/{id}"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        anime_name = anime_data['data']['title']
        # print("Data retrieved!")
        return anime_name
    else:
        raise ValueError(f"Failed to retrieve data {response.status_code}")

def get_anime_episodes(id):
    url = f"{base_url}/anime/{id}/episodes"
    response = requests.get(url)

    if response.status_code == 200:
        anime_data = response.json()
        # print("Episode data retrieved!")
        return anime_data
    else:
        raise ValueError(f"Failed to retrieve data {response.status_code}")
def get_anime_statistics(id):
    url = f"{base_url}/anime/{id}/statistics"
    response = requests.get(url)

    if response.status_code == 200:
        # print("Statistics retrieved!")
        anime_data = response.json()
        return anime_data
    else:
        raise ValueError(f"Failed to retrieve data {response.status_code}")

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
        # print("Page data retrieved!")
        return anime_data
    else:
        print(f"Failed to retrieve data {response.status_code}")
        raise ValueError()

def get_prequel(id):
    url = f"{base_url}/anime/{id}/relations"
    response = requests.get(url)

    if response.status_code == 200:
        # print("Prequel data retrieved!")
        relation_data = response.json()
        relations = [r['relation'] for r in relation_data.get('data', [])]
        return (True if "Prequel" in relations else False)
    else:
        print(f"Failed to retrieve data {response.status_code}")
        raise ValueError()


# print(json.dumps(get_anime_page(), indent=4))



# Data retrieval
def retrieve_data(page_number):
    url = f"{base_url}/anime"
    params = {"page": page_number}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        page_data = response.json()
        print(f"Data retreived for page {page_number}!")
        print(f"Starting individual data retrieval...")

        # Get ids to call other endpoints
        id_array = [anime['mal_id'] for anime in page_data.get("data", [])]
        # print("DEBUG: ID Array Finished!")
        # print(id_array)


        # Get title
        title_array = [anime['title'] for anime in page_data.get("data", [])]

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
            studios_array.append(studios)
            dem_array.append(demographics)
            themes_array.append(themes)

        # print("DEBUG: THEMES ARRAY FINISHED!")
        # print(themes_array)

        # Get age rating
        rating_array = [anime['rating'] for anime in page_data.get("data", [])]

        # Get bool if it's a sequel
        sequel_array = []
        for anime in page_data.get('data', []):
            id_num = anime['mal_id']
            sequel_array.append(get_prequel(id_num))
            time.sleep(1.1)

        # print("DEBUG: SEQUEL ARRAY FINISHED!")
        # print(sequel_array)

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

            time.sleep(1.1)

        # print("DEBUG: WC ARRAY FINISHED!")
        # print(wc_array)

        # Get metrics from "episodes" call
        forum_array = []
        for anime in page_data.get('data', []):
            id_num = anime['mal_id']
            eps_json = get_anime_episodes(id_num)

            total_replies = sum(
                ep.get('replies', 0)
                for ep in eps_json.get('data', [])
                if ep.get('mal_id', 0) <= 13
            )

            forum_array.append(total_replies)
            time.sleep(1.1)

        # print("DEBUG: FORUM ARRAY FINISHED!")
        # print(forum_array)

        types_array = [anime['type'] for anime in page_data.get("data", [])]

        # AniList GraphQL API call (might not be completely possible due to potential matching failures)

        # Check if there's a next page
        has_next = page_data.get("pagination", {}).get("has_next_page", False)

        # Final checks before proceeding
        arrays = [id_array, title_array, source_array, episode_array, synopsis_array, year_array, season_array, producers_array,
                  genres_array, studios_array, dem_array, themes_array, rating_array, sequel_array, favorites_array, scores_array,
                  wc_array, dropped_array, forum_array, types_array]

        all_same_length = all(len(lst) == len(arrays[0]) for lst in arrays)

        return (arrays if all_same_length else []), has_next
    else:
        print(f"Failed to retrieve data {response.status_code}")
        raise ValueError()

# TESTING WORKS!
# total_requests = 1000
# data_arrays, has_next, page_requests = extract_data(current_page)
# total_requests += page_requests + 1

# print(data_arrays)
# print(total_requests)

def extract_data(current_page, has_next):
    columns = [
        'mal_id', 'title', 'source', 'episodes', 'synopsis', 'year',
        'season', 'producers', 'genres', 'studios', 'demographics',
        'themes', 'rating', 'sequel', 'favorites', 'score', 'wc',
        'dropped', 'forum', 'type'
    ]

    data_frames = []

    data_arrays, has_next = retrieve_data(current_page)
        
    # Transpose lists into rows and create a temporary DataFrame
    rows = list(zip(*data_arrays))
    page_df = pd.DataFrame(rows, columns=columns)
    data_frames.append(page_df)

    return data_frames, has_next

def transform_data(df, data_frames):
    print("Starting data transformation...")
    # Concatenate incoming page data frames
    new_data = pd.concat(data_frames, ignore_index=True)
    new_data = new_data[new_data["type"] == "TV"].drop(columns="type")

    # Combine existing df with newly transformed data
    df = pd.concat([df, new_data], ignore_index=True)
    return df

def load_data(df):
    print("Starting data loading...")
    folder_path = "data/raw"
    file_name = "anime_data.csv"
    full_path = os.path.join(folder_path, file_name)
    df.to_csv(full_path, index=False)
    print(f"File successfully saved to: {full_path}")
    print(df.info())
    return df

initial_data = pd.read_csv("data/raw/anime_data.csv")

has_next = True
max_pages = 100000
page = 965

def run(has_next, current_page, df):
    print("ETL Pipline Currently Running!")
    while has_next:
        data_frames, has_next = extract_data(current_page, has_next)
        df = transform_data(df, data_frames)
        df = load_data(df)
        current_page += 1

    print("Finished! Confirmation below:")
    print(f"Current page: {current_page}")
    print(f"Has next: {has_next}")


run(has_next, page, initial_data)