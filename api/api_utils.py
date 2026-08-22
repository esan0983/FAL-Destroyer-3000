import requests
import time

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

def get_anime_episodes(id):
    url = f"{base_url}/anime/{id}/episodes"
    response = requests.get(url)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None
    
def get_anime_statistics(id):
    url = f"{base_url}/anime/{id}/statistics"
    response = requests.get(url)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None
        
    return response.json()

def get_anime_relations(id):
    url = f"{base_url}/anime/{id}/relations"
    response = requests.get(url)

    if response.status_code == 429:
        # Raise this error so the loop knows it needs to back off
        raise requests.exceptions.HTTPError("Rate limited", response=response)
        
    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None

    return response.json()

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

def get_anime(id):
    url = f"{base_url}/anime/{id}/full"
    response = requests.get(url)

    if response.status_code == 429:
        raise requests.exceptions.HTTPError("Rate limited", response=response)

    if response.status_code != 200:
        print(f"Failed to retrieve data {response.status_code}")
        return None

    return response.json()

def get_prequel(id):
    url = f"{base_url}/anime/{id}/relations"
    response = requests.get(url)

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
    response = requests.get(url, params=params)

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