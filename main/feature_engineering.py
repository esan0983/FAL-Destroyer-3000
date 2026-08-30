# main/feature_engineering.py
# Makes sure that all the data is feature-engineered and optimized for Ml accuracy and statistical analysis
# The feature engineering workflow is as follows:
# 1. Create a "drop rate" feature for statistical analysis
# 2. Z-score metrics against cohort
# 3. Drop original metrics and cohort columns
# 4. WIP: metrics will be printed so we can decide whether to drop anime that are alone in their cohort
# 5. Collapse adaptation metrics into two: take the best score from all media types and the best member count from all media types
# 6. For null adaptation score and member count, fill with sentinel value -1000
# 7. If an anime has a prequel ID that's in the database, take that ID's score metrics. If not, fill with sentinel value -10000
# 8. Drop prequel_id and mal_id columns
# 9. Remove the Award Winning genre as that's considered data leakage
# 10. Saved the pre-split dataframe for statistical analysis
# 11. Filled null ratings with an empty string and made the sequel bool into an integer
# 12. Split to training, validation, test, and inference data to avoid train-test data leakage via the next steps
# 13. Feature-engineer genre and theme columns using multi-label binarization + truncated SVD and target encoding
# 14. Performed multi-label binarization on demographics
# 15. Turned "rating" and "source" into a category type
# 16. Dropped drop rate column since it won't be used in ML
# 17. Dropped episodes since it's not recorded for inference data
# 18. Replace dead space with underscores for column names
# 19. Dropped metrics for inference data
# 20. Dropped title for non-inference data
# 21. Dropped cohort

import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def pre_split(df, init_rows):
    df['drop_rate'] = df['dropped'] / df['wc']
    
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum', 'drop_rate']
    for metric in metrics:
        if metric != "drop_rate":
            df[metric] = np.log1p(df[metric])
        df[f'{metric}_z'] = df.groupby('cohort')[metric].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    df.drop(columns=metrics)
    df.drop(columns=['cohort'])
    temp_df = df.head(init_rows)

    temp_df = temp_df.fillna({'score_z':0, 'wc_z':0, 'favorites_z':0, 'dropped_z':0, 'forum_z':0})
    df.iloc[:init_rows] = temp_df

    media_types = ['manga', 'novel', 'light_novel', 'one_shot', 'manhwa', 'manhua', 'doujinshi']
    score_cols = [media_type + "_score" for media_type in media_types]
    member_cols = [media_type + "_members" for media_type in media_types] 

    df['adaptation_score'] = df[score_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)
    df['adaptation_members'] = df[member_cols].apply(lambda row: np.nanmax(row.values) if row.notna().any() else np.nan, axis=1)
    df['adaptation_members'] = np.log1p(df['adaptation_members'])
    df = df.drop(columns=score_cols + member_cols)
    df = df.fillna({
        'adaptation_score': -1000,
        'adaptation_members': -1000
    })

    for mal_id in df['mal_id']:
        prequel_id = df.loc[df['mal_id'] == mal_id, 'prequel_id'].item()
        if not pd.isna(prequel_id) and prequel_id in df['mal_id'].values:
            df.loc[df['mal_id'] == mal_id, 'prequel_score_z'] = df.loc[df['mal_id'] == prequel_id, 'score_z'].item()
            df.loc[df['mal_id'] == mal_id, 'prequel_wc_z'] = df.loc[df['mal_id'] == prequel_id, 'wc_z'].item()
            df.loc[df['mal_id'] == mal_id, 'prequel_favorites_z'] = df.loc[df['mal_id'] == prequel_id, 'favorites_z'].item()
            df.loc[df['mal_id'] == mal_id, 'prequel_dropped_z'] = df.loc[df['mal_id'] == prequel_id, 'dropped_z'].item()
            df.loc[df['mal_id'] == mal_id, 'prequel_forum_z'] = df.loc[df['mal_id'] == prequel_id, 'forum_z'].item()
        else:
            df.loc[df['mal_id'] == mal_id, 'prequel_score_z'] = -10000
            df.loc[df['mal_id'] == mal_id, 'prequel_wc_z'] = -10000
            df.loc[df['mal_id'] == mal_id, 'prequel_favorites_z'] = -10000
            df.loc[df['mal_id'] == mal_id, 'prequel_dropped_z'] = -10000
            df.loc[df['mal_id'] == mal_id, 'prequel_forum_z'] = -10000

    df = df.drop(columns=['prequel_id', 'mal_id'])
    df['genres'] = df['genres'].apply(lambda lst: [item for item in lst if item != "Award Winning"])
    df = df.fillna({'rating': ""})
    df['sequel'] = df['sequel'].astype(int)

    return df

def splitting(df):
    inference_df = df.tail(72)
    xgb_df = df.head(3278)

    train_val_df, test_df = train_test_split(xgb_df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)

    train_df = train_df.drop(columns=['title'])
    val_df = val_df.drop(columns=['title'])
    test_df = test_df.drop(columns=['title'])

    return train_df, val_df, test_df, inference_df

def encode_multi_label_genre(df, genre_map, global_mean, metric):
    """Maps row lists to aggregated statistics of their individual genre targets."""
    means, mins, maxs = [], [], []
    
    for genres in df['genres']:
        if len(genres) == 0: 
            means.append(global_mean)
            mins.append(global_mean)
            maxs.append(global_mean)
            continue
            
        vals = [genre_map.get(g, global_mean) for g in genres]
        
        means.append(np.mean(vals))
        mins.append(np.min(vals))
        maxs.append(np.max(vals))
        
    return pd.DataFrame({
        f'genre_{metric}_mean': means,
        f'genre_{metric}_min': mins,
        f'genre_{metric}_max': maxs
    }, index=df.index)

def encode_multi_label_theme(df, theme_map, global_mean, metric):
    """Maps row lists to aggregated statistics of their individual genre targets."""
    means, mins, maxs = [], [], []
    
    for themes in df['themes']:
        if len(themes) == 0: 
            means.append(global_mean)
            mins.append(global_mean)
            maxs.append(global_mean)
            continue
            
        vals = [theme_map.get(t, global_mean) for t in themes]
        
        means.append(np.mean(vals))
        mins.append(np.min(vals))
        maxs.append(np.max(vals))
        
    return pd.DataFrame({
        f'theme_{metric}_mean': means,
        f'theme_{metric}_min': mins,
        f'theme_{metric}_max': maxs
    }, index=df.index)

def genre_mlb_svd(train_df, val_df, test_df, inference_df):
    mlb = MultiLabelBinarizer()
    train_bin = mlb.fit_transform(train_df['genres'])
    val_bin = mlb.transform(val_df['genres'])
    test_bin = mlb.transform(test_df['genres'])
    inference_bin = mlb.transform(inference_df['genres'])

    svd = TruncatedSVD(n_components=5, random_state=42)
    train_svd = svd.fit_transform(train_bin)
    val_svd = svd.transform(val_bin)
    test_svd = svd.transform(test_bin)
    inference_svd = svd.transform(inference_bin)

    svd_cols = [f'genre_svd_{i}' for i in range(5)]
    train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train_df.index)
    val_svd_df = pd.DataFrame(val_svd, columns=svd_cols, index=val_df.index)
    test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test_df.index)
    inference_svd_df = pd.DataFrame(inference_svd, columns=svd_cols, index=inference_df.index)

    return train_svd_df, val_svd_df, test_svd_df, inference_svd_df

def genre_target_encoding(train_df, val_df, test_df, inference_df, metric):

    exploded_train = train_df[['genres', metric]].explode('genres')

    global_target_mean = train_df[metric].mean()
    genre_means = exploded_train.groupby('genres')[metric].mean().to_dict()

    train_tgt_df = encode_multi_label_genre(train_df, genre_means, global_target_mean, metric)
    val_tgt_df = encode_multi_label_genre(val_df, genre_means, global_target_mean, metric)
    test_tgt_df = encode_multi_label_genre(test_df, genre_means, global_target_mean, metric)
    inference_tgt_df = encode_multi_label_genre(inference_df, genre_means, global_target_mean, metric)

    return train_tgt_df, val_tgt_df, test_tgt_df, inference_tgt_df

def theme_mlb_svd(train_df, val_df, test_df, inference_df):
    mlb = MultiLabelBinarizer()
    train_bin = mlb.fit_transform(train_df['themes'])
    val_bin = mlb.transform(val_df['themes'])
    test_bin = mlb.transform(test_df['themes'])
    inference_bin = mlb.transform(inference_df['themes'])

    svd = TruncatedSVD(n_components=7, random_state=42)
    train_svd = svd.fit_transform(train_bin)
    val_svd = svd.transform(val_bin)
    test_svd = svd.transform(test_bin)
    inference_svd = svd.transform(inference_bin)

    svd_cols = [f'theme_svd_{i}' for i in range(7)]
    train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train_df.index)
    val_svd_df = pd.DataFrame(val_svd, columns=svd_cols, index=val_df.index)
    test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test_df.index)
    inference_svd_df = pd.DataFrame(inference_svd, columns=svd_cols, index=inference_df.index)

    return train_svd_df, val_svd_df, test_svd_df, inference_svd_df

 # THEME EMBEDDINGS
def theme_target_encoding(train_df, val_df, test_df, inference_df, metric):
    exploded_train = train_df[['themes', metric]].explode('themes')

    global_target_mean = train_df[metric].mean()
    theme_means = exploded_train.groupby('themes')[metric].mean().to_dict()

    train_tgt_df = encode_multi_label_theme(train_df, theme_means, global_target_mean, metric)
    val_tgt_df = encode_multi_label_theme(val_df, theme_means, global_target_mean, metric)
    test_tgt_df = encode_multi_label_theme(test_df, theme_means, global_target_mean, metric)
    inference_tgt_df = encode_multi_label_theme(inference_df, theme_means, global_target_mean, metric)

    return train_tgt_df, val_tgt_df, test_tgt_df, inference_tgt_df

def demographic_mlb(train_df, val_df, test_df, inference_df):
    mlb = MultiLabelBinarizer()
    train_bin = mlb.fit_transform(train_df['demographics'])
    val_bin = mlb.transform(val_df['demographics'])
    test_bin = mlb.transform(test_df['demographics'])
    inference_bin = mlb.transform(inference_df['demographics'])

    def create_and_concat(df, arr):
        # Create DataFrame with MLB classes as columns and preserve original index
        mlb_df = pd.DataFrame(arr, columns=mlb.classes_, index=df.index)
        # Drop the original 'demographics' column and concat horizontally
        return pd.concat([df.drop(columns=['demographics']), mlb_df], axis=1)

    # Process all dataframes
    train_res = create_and_concat(train_df, train_bin)
    val_res = create_and_concat(val_df, val_bin)
    test_res = create_and_concat(test_df, test_bin)
    inference_res = create_and_concat(inference_df, inference_bin)

    return train_res, val_res, test_res, inference_res

def post_split(train_df, val_df, test_df, inference_df):
    metrics = ['score_z', 'wc_z', 'favorites_z', 'dropped_z', 'forum_z']

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

    for metric in metrics:
        temp_train, temp_val, temp_test, temp_inf = genre_target_encoding(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            inference_df=inference_df,
            metric=metric
        )

        train_df = train_df.join(temp_train)
        val_df = val_df.join(temp_val)
        test_df = test_df.join(temp_test)
        inference_df = inference_df.join(temp_inf)

    train_df = train_df.drop(columns=['genres'])
    val_df = val_df.drop(columns=['genres'])
    test_df = test_df.drop(columns=['genres'])
    inference_df = inference_df.drop(columns=['genres'])

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

    for metric in metrics:
        temp_train, temp_val, temp_test, temp_inf = theme_target_encoding(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            inference_df=inference_df,
            metric=metric
        )

        train_df = train_df.join(temp_train)
        val_df = val_df.join(temp_val)
        test_df = test_df.join(temp_test)
        inference_df = inference_df.join(temp_inf)

    train_df = train_df.drop(columns=['themes'])
    val_df = val_df.drop(columns=['themes'])
    test_df = test_df.drop(columns=['themes'])
    inference_df = inference_df.drop(columns=['themes'])

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

    processed_dfs = {}

    for name, df in [('train', train_df), ('val', val_df), ('test', test_df), ('inference', inference_df)]:
        df['rating'] = df['rating'].astype('category')
        df['source'] = df['source'].astype('category')
        
        df = df.drop(columns=['drop_rate', 'episodes', 'drop_rate_z', 'cohort'])
        df.columns = df.columns.str.replace(' ', '_')

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

    train_df, val_df, test_df, inference_df = splitting(df)
    train_df, val_df, test_df, inference_df = post_split(train_df, val_df, test_df, inference_df)

    print("Saving ML DFs...")
    train_df.to_parquet("data/ml_data/train_df.parquet", engine="pyarrow")
    val_df.to_parquet("data/ml_data/val_df.parquet", engine="pyarrow")
    test_df.to_parquet("data/ml_data/test_df.parquet", engine="pyarrow")
    inference_df.to_parquet("data/ml_data/inference_df.parquet", engine="pyarrow")