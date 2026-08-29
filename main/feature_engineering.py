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
# 11. Split to training, validation, test, and inference data to avoid train-test data leakage via the next steps
# 12. Feature-engineer genre and theme columns using multi-label binarization + truncated SVD and target encoding
# 13. Performed multi-label binarization on demographics
# 14. Turned "rating" and "source" into a category type
# 15. Dropped drop rate column since it won't be used in ML

import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def pre_split(df, init_rows):
    df['drop_rate'] = df['dropped'] / df['wc']
    assert np.isfinite(df['drop_rate']).all()
    
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum', 'drop_rate']
    for metric in metrics:
        if metric is not "drop_rate":
            df[metric] = np.log1p(df[metric])
        df[f'{metric}_z'] = df.groupby('cohort')[metric].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    df.drop(columns=metrics)
    df.drop(columns=['cohort'])
    temp_df = df.head(init_rows)
    print(temp_df[[f"{metric}_z" for metric in metrics]].isna().sum()) # decide at this point whether to impute or drop

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
            df.loc(df['mal_id'] == mal_id, 'prequel_score_z') = df.loc[df['mal_id'] == prequel_id, 'score_z'].item()
            df.loc(df['mal_id'] == mal_id, 'prequel_wc_z') = df.loc[df['mal_id'] == prequel_id, 'wc_z'].item()
            df.loc(df['mal_id'] == mal_id, 'prequel_favorites_z') = df.loc[df['mal_id'] == prequel_id, 'favorites_z'].item()
            df.loc(df['mal_id'] == mal_id, 'prequel_dropped_z') = df.loc[df['mal_id'] == prequel_id, 'dropped_z'].item()
            df.loc(df['mal_id'] == mal_id, 'prequel_forum_z') = df.loc[df['mal_id'] == prequel_id, 'forum_z'].item()
        else:
            df.loc(df['mal_id'] == mal_id, 'prequel_score_z') = -10000
            df.loc(df['mal_id'] == mal_id, 'prequel_wc_z') = -10000
            df.loc(df['mal_id'] == mal_id, 'prequel_favorites_z') = -10000
            df.loc(df['mal_id'] == mal_id, 'prequel_dropped_z') = -10000
            df.loc(df['mal_id'] == mal_id, 'prequel_forum_z') = -10000

    df = df.drop(columns=['prequel_id', 'mal_id'])
    df['genres'] = df['genres'].apply(lambda lst: [item for item in lst if item != "Award Winning"])

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

def encode_multi_label_target(df, genre_map, global_mean):
    """Maps row lists to aggregated statistics of their individual genre targets."""
    means, mins, maxs = [], [], []
    
    for genres in df['genre']:
        if not genres: 
            means.append(global_mean)
            mins.append(global_mean)
            maxs.append(global_mean)
            continue
            
        vals = [genre_map.get(g, global_mean) for g in genres]
        
        means.append(np.mean(vals))
        mins.append(np.min(vals))
        maxs.append(np.max(vals))
        
    return pd.DataFrame({
        'genre_tgt_mean': means,
        'genre_tgt_min': mins,
        'genre_tgt_max': maxs
    }, index=df.index)

# GENRE EMBEDDINGS
def genre_embedding(train_df, val_df, test_df, inference_df, metric):
    # NOTE TO FUTURE ME: YOU'RE DOING MLB FIVE TIMES. YOU ONLY NEED TO DO THAT FOR TARGET ENCODING, SPLIT THIS INTO MLB/SVD AND TARGET ENCODING FUNCTIONS
    mlb = MultiLabelBinarizer()
    train_bin = mlb.fit_transform(train_df['genre'])
    val_bin = mlb.transform(val_df['genre'])
    test_bin = mlb.transform(test_df['genre'])
    inference_bin = mlb.transform(inference_df['genre'])

    svd = TruncatedSVD(n_components=5, random_state=42)
    train_svd = svd.fit_transform(train_bin)
    val_svd = svd.transform(val_bin)
    test_svd = svd.transform(test_bin)
    inference_svd = svd.transform(inference_bin)

    svd_cols = [f'{metric}_genre_svd_{i}' for i in range(5)]
    train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train_df.index)
    val_svd_df = pd.DataFrame(val_svd, columns=svd_cols, index=val_df.index)
    test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test_df.index)
    inference_svd_df = pd.DataFrame(inference_svd, columns=svd_cols, index=inference_df.index)

    exploded_train = train_df[['genre', metric]].explode('genre')

    global_target_mean = train_df[metric].mean()
    genre_means = exploded_train.groupby('genre')[metric].mean().to_dict()

    # NOTE TO FUTURE ME: ADD A METRIC PARAMETER SO COLUMN NAMES ARE NAMED PROPERLY
    train_tgt_df = encode_multi_label_target(train_df, genre_means, global_target_mean)
    val_tgt_df = encode_multi_label_target(val_df, genre_means, global_target_mean)
    test_tgt_df = encode_multi_label_target(test_df, genre_means, global_target_mean)
    inference_tgt_df = encode_multi_label_target(inference_df, genre_means, global_target_mean)

    X_train_genre = pd.concat([train_tgt_df, train_svd_df], axis=1)
    X_val_genre = pd.concat([val_tgt_df, val_svd_df], axis=1)
    X_test_genre = pd.concat([test_tgt_df, test_svd_df], axis=1)
    X_inference_genre = pd.concat([inference_tgt_df, inference_svd_df], axis=1)

    return X_train_genre, X_val_genre, X_test_genre, X_inference_genre

 # THEME EMBEDDINGS
def theme_embedding(train_df, val_df, test_df, inference_df, metric):
    # NOTE TO FUTURE ME: YOU'RE DOING MLB FIVE TIMES. YOU ONLY NEED TO DO THAT FOR TARGET ENCODING, SPLIT THIS INTO MLB/SVD AND TARGET ENCODING FUNCTIONS
    mlb = MultiLabelBinarizer()
    train_bin = mlb.fit_transform(train_df['theme'])
    val_bin = mlb.transform(val_df['theme'])
    test_bin = mlb.transform(test_df['theme'])
    inference_bin = mlb.transform(inference_df['theme'])

    svd = TruncatedSVD(n_components=7, random_state=42)
    train_svd = svd.fit_transform(train_bin)
    val_svd = svd.transform(val_bin)
    test_svd = svd.transform(test_bin)
    inference_svd = svd.transform(inference_bin)

    svd_cols = [f'{metric}_theme_svd_{i}' for i in range(7)]
    train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train_df.index)
    val_svd_df = pd.DataFrame(val_svd, columns=svd_cols, index=val_df.index)
    test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test_df.index)
    inference_svd_df = pd.DataFrame(inference_svd, columns=svd_cols, index=inference_df.index)

    exploded_train = train_df[['theme', metric]].explode('genre')

    global_target_mean = train_df[metric].mean()
    theme_means = exploded_train.groupby('theme')[metric].mean().to_dict()

    # FOR FUTURE ME: ADD A METRIC PARAMETER SO COLUMN NAMES ARE NAMED PROPERLY
    train_tgt_df = encode_multi_label_target(train_df, theme_means, global_target_mean)
    val_tgt_df = encode_multi_label_target(val_df, theme_means, global_target_mean)
    test_tgt_df = encode_multi_label_target(test_df, theme_means, global_target_mean)
    inference_tgt_df = encode_multi_label_target(inference_df, theme_means, global_target_mean)

    X_train_theme = pd.concat([train_tgt_df, train_svd_df], axis=1)
    X_val_theme = pd.concat([val_tgt_df, val_svd_df], axis=1)
    X_test_theme = pd.concat([test_tgt_df, test_svd_df], axis=1)
    X_inference_theme = pd.concat([inference_tgt_df, inference_svd_df], axis=1)

    return X_train_theme, X_val_theme, X_test_theme, X_inference_theme

# DEMOGRAPHICS EMBEDDINGS (not really an embedding, it only does MLB)
def demographic_embedding(train_df, val_df, test_df, inference_df):
    # NOTE TO FUTURE ME: YOU'RE DOING MLB FIVE TIMES IN THE LOOP.
    mlb = MultiLabelBinarizer()
    train_bin = mlb.fit_transform(train_df['demographic'])
    val_bin = mlb.transform(val_df['demographic'])
    test_bin = mlb.transform(test_df['demographic'])
    inference_bin = mlb.transform(inference_df['demographic'])

    return train_bin, val_bin, test_bin, inference_bin

def post_split(train_df, val_df, test_df, inference_df):
    metrics = ['score_z', 'wc_z', 'favorites_z', 'dropped_z', 'forum_z']

    # NOTE FOR FUTURE ME: THESE LOOPS WILL BE CHANGED ONCE I SPLIT THE EMBEDDING FUNCTIONS INTO MLB/SVD AND TARGET ENCODING
    for metric in metrics:
        temp_train, temp_val, temp_test, temp_inf = genre_embedding(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            inference_df=inference_df,
            metric=metric
        )

        train_df = train_df.drop(columns=['genre']).join(temp_train)
        val_df = val_df.drop(columns=['genre']).join(temp_val)
        test_df = test_df.drop(columns=['genre']).join(temp_test)
        inference_df = inference_df.drop(columns=['genre']).join(temp_inf)

   
    for metric in metrics:
        temp_train, temp_val, temp_test, temp_inf = theme_embedding(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            inference_df=inference_df,
            metric=metric
        )

        train_df = train_df.drop(columns=['theme']).join(temp_train)
        val_df = val_df.drop(columns=['theme']).join(temp_val)
        test_df = test_df.drop(columns=['theme']).join(temp_test)
        inference_df = inference_df.drop(columns=['theme']).join(temp_inf)

        
    for metric in metrics:
        temp_train, temp_val, temp_test, temp_inf = demographic_embedding(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            inference_df=inference_df
        )

        train_df = train_df.drop(columns=['demographics']).join(temp_train)
        val_df = val_df.drop(columns=['demographics']).join(temp_val)
        test_df = test_df.drop(columns=['demographics']).join(temp_test)
        inference_df = inference_df.drop(columns=['demographics']).join(temp_inf)

    for df in [train_df, val_df, test_df, inference_df]:
        df['rating'] = df['rating'].astype('category')
        df['source'] = df['rating'].astype('category')

    df.drop(columns=['drop_rate'])

    return train_df, val_df, test_df, inference_df

if __name__ == "__main__":
    initial_df = pd.read_parquet("data/processed/anime_data_1.parquet")
    current_df = pd.read_parquet("data/processed/real_data_1.parquet")
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