# api/api_utils.py
# api calling functions

from bs4 import BeautifulSoup
import requests
import time
import random

base_url = "https://api.tenrai.org/v1"

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

    return response.json()

def get_anime_episodes(id):
    url = f"{base_url}/anime/{id}/episodes"
    success = False
    while not success:
        try:
            response = requests.get(url)
            success = True
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None

    return response.json()
    
def get_anime_statistics(id):
    url = f"{base_url}/anime/{id}/statistics"
    success = False
    while not success:
        try:
            response = requests.get(url)
            success = True
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None
        
    return response.json()

def get_anime_relations(id):
    url = f"{base_url}/anime/{id}/relations"
    success = False
    while not success:
        try:
            response = requests.get(url)
            success = True
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None

    return response.json()

def get_manga(id):
    url = f"{base_url}/manga/{id}"
    success = False
    while not success:
        try:
            response = requests.get(url)
            success = True
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None
        
    return response.json()

def get_anime(id):
    url = f"{base_url}/anime/{id}/full"

    success = False
    while not success:
        try:
            response = requests.get(url)
            success = True
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)

    if response.status_code == 429:
        raise requests.exceptions.HTTPError("Rate limited", response=response)

    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None

    return response.json()

def get_prequel(id):
    url = f"{base_url}/anime/{id}/relations"
    success = False
    while not success:
        try:
            response = requests.get(url)
            success = True
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)

    if response.status_code == 200:
        # print("Prequel data retrieved!")
        relation_data = response.json()
        relations = [r['relation'] for r in relation_data.get('data', [])]
        prequel = next((item for item in relation_data["data"] if item["relation"] == "Prequel"), None)
        return_id = None
        if prequel is not None:
            return_id = prequel["entry"][0]["mal_id"] if prequel and prequel["entry"] else None
        return (True if "Prequel" in relations else False), return_id
    else:
        print(f"Failed to retrieve data {response.status_code}")
        raise ValueError()

def get_season(params):
    url = f"{base_url}/seasons/2026/fall"
    success = False
    while not success:
        try:
            response = requests.get(url)
            success = True
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)

    if response.status_code == 429:
            raise requests.exceptions.HTTPError("Rate limited", response=response)
    
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None

    return response.json()

def get_ids():
    mal_ids = []
    page_number = 1
    has_next = True
    while has_next:
        params = {"page": page_number}
        time.sleep(0.34)
        page_success = False

        while not page_success:
            try:
                page_list = get_season(params)
                print(f"Got page {page_number}!")
                page_success = True
            except Exception as e:
                print(f"get_season failed for {page_number}: {e}")
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    print(f"Rate limited on ID {page_number}. Backing off 5s and retrying...")
                    time.sleep(5)
                else:
                    break

        page_data = page_list.get('data', {})
        for data in page_data:
            if data.get('type', "") == "TV":
                mal_ids.append(data.get('mal_id', None))

        has_next = page_list.get('pagination', {}).get('has_next_page', False)
        page_number += 1
    return mal_ids

def get_unique_users(link: str) -> int:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    unique_users = set()
    offset = 0
    page_step = 50
    seen_offsets = set()
    attempt = 1

    while True:
        page_url = link if offset == 0 else f"{link}&show={offset}"
        try:
            response = requests.get(page_url, headers=headers)
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = random.uniform(0, 1)
            print(f"Connection error at offset {offset} ({type(e).__name__}). Backing off {wait:.1f}s and retrying...")
            time.sleep(wait)
            continue

        if "Just a moment" in response.text or "cf-challenge" in response.text or response.status_code == 403:
            print(f"Possible Cloudflare block at offset {offset}. Stopping and backing off.")
            wait = (3 ** attempt) + random.uniform(0, 1)
            attempt += 1
            time.sleep(wait)
            continue

        attempt = 1

        if response.status_code in [408, 502, 504]:
            print(f"Your wifi cut out. Will retry.")
            time.sleep(5)
            continue

        if response.status_code != 200:
            print(f"Failed to fetch page offset {offset}: Status {response.status_code}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        post_containers = soup.select("div.forum-topic-message.message")
        page_users = set()
        for post in post_containers:
            username = post.get("data-user")
            if username:
                page_users.add(username)

        if not page_users:
            print(f"No users found at offset {offset}. Stopping")
            break

        if offset in seen_offsets:
            break
        seen_offsets.add(offset)

        previous_total = len(unique_users)
        unique_users.update(page_users)
        new_added = len(unique_users) - previous_total

        print(
            f"Page offset {offset}: Found {len(page_users)} user(s) on this page ({new_added} new unique users)."
        )

        next_offset = offset + page_step
        next_page_link = soup.select_one(f'a[href*="show={next_offset}"]')

        if not next_page_link:
            print(f"No link found for offset {next_offset}, so we reached the last page.")
            break

        offset = next_offset
        time.sleep(random.uniform(2, 3))
    return len(unique_users)