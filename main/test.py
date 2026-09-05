import pandas as pd

old_data = pd.read_parquet("data/processed/anime_data_1.parquet")
old_ids = set(old_data['mal_id'].tolist())

new_data = pd.read_csv("data/raw/anime_data.csv")
new_ids = set(new_data['mal_id'].tolist())

print(old_ids - new_ids)