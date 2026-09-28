# main/handle_nan.py
# ElasticNet can't accept NaNs the way RF/XGBoost can, so the sentinel-value
# trick in feature_engineering.py (-1000 / -10000 for "no adaptation" /
# "no prequel") won't work here: ElasticNet is scale-sensitive, and a sentinel
# that far outside the real distribution would dominate the fit rather than
# just letting a tree split it into its own branch.
#
# Instead: for each column where NaN is structural ("this anime has no
# adaptation" / "no prequel"), add a binary <col>_missing indicator so that
# information isn't thrown away, then fill the NaN with the median computed
# on the training data only, to keep the same train-fit / transform-elsewhere
# discipline as the rest of the pipeline (OneHotEncoder, cohort z-scores, etc).
#
# Median (not mean) because these columns are heavily skewed (adaptation_members
# especially) and the fitted value is only ever a placeholder for "unknown" --
# it shouldn't be pulled around by outliers.

import pandas as pd

# Columns where NaN means "this doesn't exist for this anime" rather than
# "value unknown but present" -- confirmed via adaptation_score/adaptation_members
# missingness not being fully redundant (68 rows missing at least one of the
# two, only 42 missing both), so each column gets its own indicator rather
# than one combined "has_adaptation" flag.
NAN_INDICATOR_COLUMNS = [
    "adaptation_score",
    "adaptation_members",
    "prequel_score",
    "prequel_wc",
]


def handle_missing_values(train_df, other_df=None, columns=None):
    """
    Adds a <col>_missing indicator and median-imputes NaNs for the given
    columns (defaults to NAN_INDICATOR_COLUMNS), fitting the median on
    train_df only.

    Mirrors the signature of multivalue_preprocessing/encode_features so it
    slots into the same fit-on-train, transform-elsewhere pattern used
    elsewhere in the pipeline: call it on (train, val), then again on
    (train, test), then again on (train, inference), always passing the
    original train_df so the imputation values stay consistent.

    Returns (train_out, other_out) if other_df is given, else train_out.
    """
    if columns is None:
        columns = [c for c in NAN_INDICATOR_COLUMNS if c in train_df.columns]

    train_out = train_df.copy()
    other_out = other_df.copy() if other_df is not None else None

    for col in columns:
        if col not in train_out.columns:
            continue

        # Cast nullable extension dtypes (e.g. Int64 for prequel_wc) to plain
        # float64 up front -- fillna() with a float median fails against an
        # Int64 column ("Invalid value '...' for dtype 'Int64'") since Int64
        # can only hold whole numbers or pd.NA.
        if pd.api.types.is_extension_array_dtype(train_out[col]):
            train_out[col] = train_out[col].astype("float64")
        if other_out is not None and col in other_out.columns and pd.api.types.is_extension_array_dtype(other_out[col]):
            other_out[col] = other_out[col].astype("float64")

        train_out[f"{col}_missing"] = train_out[col].isna().astype(int)
        if other_out is not None and col in other_out.columns:
            other_out[f"{col}_missing"] = other_out[col].isna().astype(int)

        median_val = train_out[col].median()

        train_out[col] = train_out[col].fillna(median_val)
        if other_out is not None and col in other_out.columns:
            other_out[col] = other_out[col].fillna(median_val)

    if other_df is not None:
        return train_out, other_out
    return train_out


if __name__ == "__main__":
    # One-off run against the uploaded train_df.parquet: impute in place
    # (train-only, so the median is fit on itself) and report what changed.
    df = pd.read_parquet("/mnt/user-data/uploads/train_df.parquet")

    print("Before:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())

    df_imputed = handle_missing_values(df)

    print("\nAfter:")
    na_after = df_imputed.isna().sum()
    print(na_after[na_after > 0].to_string() if na_after.sum() > 0 else "No NaNs remaining.")

    print("\nNew indicator columns:")
    print([c for c in df_imputed.columns if c.endswith("_missing")])

    df_imputed.to_parquet("/mnt/user-data/outputs/train_df.parquet", index=False)
    print("\nSaved imputed file to /mnt/user-data/outputs/train_df.parquet")