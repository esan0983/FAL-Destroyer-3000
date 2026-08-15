import pandas as pd, os
df = pd.read_csv("../data/processed/anime_data_1.csv")
have_images = set(int(f.split('.')[0]) for f in os.listdir("../data/images") if f.endswith('.jpg'))
missing = df[~df['mal_id'].isin(have_images)]
print(f"{len(missing)} / {len(df)} missing thumbnails")