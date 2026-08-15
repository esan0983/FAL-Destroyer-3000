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
        # print("Page data retrieved!")
        return anime_data
    else:
        print(f"Failed to retrieve data {response.status_code}")
        raise ValueError()

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

def get_anime_relations(id):
    url = f"{base_url}/anime/{id}/relations"
    response = requests.get(url)

    if response.status_code == 200:
        # print("Statistics retrieved!")
        anime_data = response.json()
        return anime_data
    else:
        raise ValueError(f"Failed to retrieve data {response.status_code}")

def get_manga(id):
    url = f"{base_url}/manga/{id}"
    response = requests.get(url)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None
        
    return response.json()