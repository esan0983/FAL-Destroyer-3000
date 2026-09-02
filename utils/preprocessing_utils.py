# utils/preprocessing_utils.py
# Functions used in the feature engineering phase.

import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

# Not used
# Performs target encoding for genres
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

# Not used
# Performs target encoding for themes
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

# Performs multi-label binarization + truncated SVD to encode genres without bloating dimension count
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

# Performs multi-label binarization + truncated SVD to encode themes without bloating dimension count
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

# Performs multi-label binarization on demographics
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