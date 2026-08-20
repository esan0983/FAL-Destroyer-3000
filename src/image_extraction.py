# DISCLAIMER: This will be separate from etl2.py since I want to emphasize that the ETL phase loads the dataset to be used in EDA and
# statistical analysis.

import os
import requests
import pandas as pd
import json
import time

from utils import base_url, get_anime

def download_image(image_url, folder_path, file_name):
    local_filename = os.path.join(folder_path, file_name)

    os.makedirs(folder_path, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(image_url, headers=headers)

    if response.status_code == 200:
        with open(local_filename, "wb") as file:
            file.write(response.content)
        print(f"Image successfully saved to: {local_filename}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

import time

def run(df, folder_path, start_id):
    for anime_id in df['mal_id']:
        if anime_id > start_id:
            # This flag controls the retry behavior for the current ID
            success = False
            
            while not success:
                try:
                    anime_json = get_anime(anime_id)
                    success = True 
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        print(f"Rate limited on id {anime_id}. Backing off 5s and retrying...")
                        time.sleep(5)
                    else:
                        print(f"Unhandled HTTP error on id {anime_id}: {e}. Will return none on the json.")

            if anime_json is None:
                return None
            
            print(f"Collected ID {anime_id}")

            image_url = anime_json['data']['images']['jpg']['large_image_url']
            file_name = str(anime_id) + ".jpg"

            download_image(image_url, folder_path, file_name)


if __name__ == "__main__":
    initial_data = pd.read_csv("data/processed/anime_data_1.csv")
    folder_path = "data/images"

    STARTING_ID = 0 # start AFTER STARTING_ID


    run(initial_data, folder_path, STARTING_ID)