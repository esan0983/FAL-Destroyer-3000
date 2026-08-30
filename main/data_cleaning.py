# main/data_cleaning.py
# Cleans data scraped by etl2.py. Should be used for both pre-FAL data (anime_data.csv) and FAL data (current_data.csv)
# The data cleaning workflow:
# 1. Removes duplicates
# 2. Turns multi-valued columns into lists (they were not proper lists when scraped)
# 3. Prints which columns have nulls and how many
# 4. Drops the thumbnail column that existed due to some CSV error
# 5. Calls describe for both datasets for potential workflow changes and bug fixes

import pandas as pd
from utils import parse_list_col

def clean(df):
    df = df.drop_duplicates(subset=['mal_id'], keep='first')

    lists = ['genres', 'demographics', 'themes']
    df[lists] = df[lists].map(parse_list_col)

    print("DF NaNs:")
    print(df.isna().sum())

    df = df.drop(columns=['thumbnail'])
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/anime_data.csv")
    current_df = pd.read_csv("data/raw/current_data.csv")

    print("Cleaning pre-FAL DF...")
    df = clean(df)
    print(df.describe(include='all'))

    print("Cleaning FAL DF...")
    current_df = clean(current_df)
    print(current_df.describe(include='all'))
    
    df.to_parquet("data/processed/anime_data_1.parquet", engine="pyarrow")
    current_df.to_parquet("data/processed/current_data_1.parquet", engine="pyarrow")
    print("DFs saved!")
