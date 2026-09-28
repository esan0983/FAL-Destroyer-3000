import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import demographic_mlb, parse_list_col, symlog_transform

def pre_split(df):
    df['demographics'] = df['demographics'].apply(parse_list_col)
    df['genres'] = df['genres'].apply(parse_list_col)
    df['themes'] = df['themes'].apply(parse_list_col)


    media_types = ['manga', 'novel', 'light_novel', 'one_shot', 'manhwa', 'manhua', 'doujinshi']
    score_cols = [media_type + "_score" for media_type in media_types]
    member_cols = [media_type + "_members" for media_type in media_types] 
    df['adaptation_score'] = df[score_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)
    df['adaptation_members'] = df[member_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)
    df = df.drop(columns=score_cols + member_cols)

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