# main/feature_engineering.py
# Makes sure that all the data is feature-engineered and optimized for Ml accuracy and statistical analysis
# Preprocessing flow not finalized yet

import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from utils import (
    genre_mlb_svd,
    theme_mlb_svd,
    demographic_mlb
)

def pre_split(df):
    df['drop_rate'] = df['dropped'] / df['wc']

    # to handle skewness
    df[['wc', 'favorites', 'dropped', 'forum']] = np.log1p(df[['wc', 'favorites', 'dropped', 'forum']])

    # collapsing adaptation stats
    media_types = ['manga', 'novel', 'light_novel', 'one_shot', 'manhwa', 'manhua', 'doujinshi']
    score_cols = [media_type + "_score" for media_type in media_types]
    member_cols = [media_type + "_members" for media_type in media_types] 
    df['adaptation_score'] = df[score_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)
    df['adaptation_members'] = df[member_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)
    df['adaptation_members'] = np.log1p(df['adaptation_members'])
    df = df.drop(columns=score_cols + member_cols)

    # split cohort into two: season and year
    df[['season', 'year']] = df['cohort'].str.split(" ", expand=True)
    df = df.drop(columns=['cohort'])
    df['year'] = df['year'].astype(int)

    # handle prequel data
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum', 'year', 'season']
    mapping_df = df[['mal_id'] + metrics].drop_duplicates(subset=['mal_id']).set_index('mal_id')
    for metric in metrics:
        df[f"prequel_{metric}"] = df['prequel_id'].map(mapping_df[metric])

    df = df.drop(columns=['prequel_id'])

    # including the Award Winning genre is considered data leakage
    df['genres'] = df['genres'].apply(lambda lst: [item for item in lst if item != "Award Winning"])
    
    df = df.fillna({'rating': ""})

    df['sequel'] = df['sequel'].astype(int)

    df['rating'] = df['rating'].astype('category')

    df['source'] = df['source'].astype('category')

    df['season'] = df['season'].astype('category')

    df['prequel_season'] = df['prequel_season'].astype('category')

    df = df.drop(columns=['episodes', 'mal_id', 'sequel'])

    return df

def splitting(df, init_rows):
    inference_df = df.tail(len(df) - init_rows)
    xgb_df = df.head(init_rows)

    train_val_df, test_df = train_test_split(xgb_df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)

    train_df = train_df.drop(columns=['title'])
    val_df = val_df.drop(columns=['title'])
    test_df = test_df.drop(columns=['title'])

    return train_df, val_df, test_df, inference_df

def post_split(train_df, val_df, test_df, inference_df):
    # GENRES
    temp_train, temp_val, temp_test, temp_inf = genre_mlb_svd(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        inference_df=inference_df
    )

    train_df = train_df.join(temp_train)
    val_df = val_df.join(temp_val)
    test_df = test_df.join(temp_test)
    inference_df = inference_df.join(temp_inf)

    # THEMES
    temp_train, temp_val, temp_test, temp_inf = theme_mlb_svd(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        inference_df=inference_df
    )

    train_df = train_df.join(temp_train)
    val_df = val_df.join(temp_val)
    test_df = test_df.join(temp_test)
    inference_df = inference_df.join(temp_inf)

    # DEMOGRAPHICS
    temp_train, temp_val, temp_test, temp_inf = demographic_mlb(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        inference_df=inference_df
    )

    train_df = temp_train
    val_df = temp_val
    test_df = temp_test
    inference_df = temp_inf

    # removing unnecessary columns for inference and non-inference
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']

    processed_dfs = {}

    for name, df in [('train', train_df), ('val', val_df), ('test', test_df), ('inference', inference_df)]:
        df.columns = df.columns.str.replace(' ', '_')
        df = df.drop(columns=['drop_rate', 'genres', 'themes']) # WILL DITCH TARGET ENCODING FOR NOW

        if name == 'inference':
            df = df.drop(columns=[metric for metric in metrics if metric in df.columns])
        else:
            df = df.drop(columns=['title'], errors='ignore')
            
        processed_dfs[name] = df

    train_df, val_df, test_df, inference_df = processed_dfs['train'], processed_dfs['val'], processed_dfs['test'], processed_dfs['inference']

    return train_df, val_df, test_df, inference_df

if __name__ == "__main__":
    initial_df = pd.read_parquet("data/processed/anime_data_1.parquet")
    current_df = pd.read_parquet("data/processed/current_data_1.parquet")
    init_rows = len(initial_df)
    df = pd.concat([initial_df, current_df], axis=0)

    df = pre_split(df, init_rows)
    stats_df = df.head(init_rows)
    print("Saving stats_df...")
    stats_df.to_parquet("data/processed/stats_df.parquet", engine="pyarrow")

    train_df, val_df, test_df, inference_df = splitting(df, init_rows)
    train_df, val_df, test_df, inference_df = post_split(train_df, val_df, test_df, inference_df)

    print("Saving ML DFs...")
    train_df.to_parquet("data/ml_data/train_df.parquet", engine="pyarrow")
    val_df.to_parquet("data/ml_data/val_df.parquet", engine="pyarrow")
    test_df.to_parquet("data/ml_data/test_df.parquet", engine="pyarrow")
    inference_df.to_parquet("data/ml_data/inference_df.parquet", engine="pyarrow")