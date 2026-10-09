import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import demographic_mlb, parse_list_col

def pick_years(primary, tiebreak, start_years, end_years):
    """
    All inputs are (n_rows, n_media_types) float arrays. For each row, picks the media
    type with the highest `primary` value (ties broken by `tiebreak`, then by column
    order) and returns that type's start and end year. Rows with no `primary` value
    at all get NaN. A year of 0 is the placeholder collect_adaptations uses for a
    missing year, so it also becomes NaN.
    """
    has_primary = ~np.isnan(primary).all(axis=1)
    primary_filled = np.where(np.isnan(primary), -np.inf, primary)
    is_best = primary_filled == primary_filled.max(axis=1, keepdims=True)
    tiebreak_filled = np.where(np.isnan(tiebreak), -1.0, tiebreak)
    best_idx = np.where(is_best, tiebreak_filled, -np.inf).argmax(axis=1)

    rows = np.arange(len(primary))
    picked_start = start_years[rows, best_idx].copy()
    picked_end = end_years[rows, best_idx].copy()

    for picked in (picked_start, picked_end):
        picked[~has_primary | (picked == 0)] = np.nan

    return picked_start, picked_end

def pre_split(df):
    # current_data.csv comes straight from read_csv, so these columns are strings like
    # "['Shounen', 'Seinen']" there, while the processed parquets already hold real lists.
    # Iterating over a string yields single characters, which is where the warning came from.
    lists = ['genres', 'demographics', 'themes']
    df[lists] = df[lists].map(parse_list_col)

    media_types = ['manga', 'novel', 'light_novel', 'one_shot', 'manhwa', 'manhua', 'doujinshi']
    score_cols = [media_type + "_score" for media_type in media_types]
    member_cols = [media_type + "_members" for media_type in media_types]
    start_cols = [media_type + "_start_year" for media_type in media_types]
    end_cols = [media_type + "_end_year" for media_type in media_types]

    df['adaptation_score'] = df[score_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)
    df['adaptation_members'] = df[member_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)

    # Start/end years of the best source material, picked two ways so member count
    # doesn't dominate: by highest score and by highest member count.
    scores = df[score_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    members = df[member_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    starts = df[start_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    ends = df[end_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

    # by score; ties go to the source with more members
    df['adaptation_score_start_year'], df['adaptation_score_end_year'] = pick_years(
        primary=scores, tiebreak=members, start_years=starts, end_years=ends
    )
    # by members; ties go to the source with the higher score
    df['adaptation_members_start_year'], df['adaptation_members_end_year'] = pick_years(
        primary=members, tiebreak=scores, start_years=starts, end_years=ends
    )

    df = df.drop(columns=score_cols + member_cols + start_cols + end_cols)

    df[['season', 'year']] = df['cohort'].str.split(" ", expand=True)
    df = df.drop(columns=['cohort'])
    df['year'] = df['year'].astype(int)

    metrics = ['score', 'wc']
    mapping_df = df[['mal_id'] + metrics].drop_duplicates(subset=['mal_id']).set_index('mal_id')
    for metric in metrics:
        df[f"prequel_{metric}"] = df['prequel_id'].map(mapping_df[metric])

    df = df.drop(columns=['prequel_id', 'score', 'wc', 'mal_id'])

    df['genres'] = df['genres'].apply(lambda lst: [item for item in lst if item != "Award Winning"])
        
    df = df.fillna({'rating': ""})

    df['rating'] = df['rating'].astype('category')

    df['source'] = df['source'].astype('category')

    return df

def split(df, init_rows):
    inference_df = df.tail(len(df) - init_rows)
    xgb_df = df.head(init_rows)

    train_df, test_df = train_test_split(xgb_df, test_size=0.15, random_state=42)

    train_df = train_df.drop(columns=['anime'])
    test_df = test_df.drop(columns=['anime'])

    return train_df, test_df, inference_df

def post_split(train_df,  test_df, inference_df):
    temp_train, temp_test, temp_inf = demographic_mlb(
            train_df=train_df,
            test_df=test_df,
            inference_df=inference_df
        )
    
    train_df = temp_train
    test_df = temp_test
    inference_df = temp_inf

    # removing unnecessary columns for inference and non-inference
    metrics = ['points']

    processed_dfs = {}

    for name, df in [('train', train_df), ('test', test_df), ('inference', inference_df)]:
        df.columns = df.columns.str.replace(' ', '_')

        if name == 'inference':
            df = df.drop(columns=[metric for metric in metrics if metric in df.columns])
        else:
            df = df.drop(columns=['title'], errors='ignore')
            
        processed_dfs[name] = df

    train_df, test_df, inference_df = processed_dfs['train'], processed_dfs['test'], processed_dfs['inference']

    return train_df, test_df, inference_df

def main():
    current_df = pd.read_csv("data/raw/current_data.csv")
    forecasting_df = pd.read_parquet("data/forecasting/processed/week1_processed.parquet")

    missing_cols = set(current_df.columns) - set(forecasting_df.columns)
    missing_cols = list(missing_cols)

    current_df = current_df.drop(columns=missing_cols)

    print(current_df.info())

    for i in range(13):
        new_current_df = current_df.copy()
        forecasting_df = pd.read_parquet(f"data/forecasting/processed/week{i+1}_processed.parquet")

        init_rows = len(forecasting_df)
        df = pd.concat([forecasting_df, new_current_df], axis=0)

        df = pre_split(df)

        train_df, test_df, inference_df = split(df, init_rows)
        train_df, test_df, inference_df = post_split(train_df, test_df, inference_df)

        train_path = Path(f"data/forecasting/ml_data/week{i+1}/train_df.parquet")
        test_path = Path(f"data/forecasting/ml_data/week{i+1}/test_df.parquet")
        inference_path = Path(f"data/forecasting/ml_data/week{i+1}/inference_df.parquet")

        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        inference_path.parent.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(train_path, engine="pyarrow")
        test_df.to_parquet(test_path, engine="pyarrow")
        inference_df.to_parquet(inference_path, engine="pyarrow")

        print(f"Week {i+1} DFs saved!")

if __name__ == "__main__":
    main()


# FOR THIS TO BE USABLE, INTEGRATE INFERENCE DATA TO ALL PARQUETS