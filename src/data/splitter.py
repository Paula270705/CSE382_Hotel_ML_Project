import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_and_save_splits(df: pd.DataFrame, splits_dir: str = "data/splits", seed: int = 42):
    os.makedirs(splits_dir, exist_ok=True)

    # استبعاد الحجوزات التالفة التي تحتوي على 0 ضيوف (إجمالي البالغين والأطفال والرضع = 0)
    total_guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    df_valid = df.loc[total_guests > 0].copy().reset_index(drop=True)

    # تقسيم طبقي (Stratified) حسب حالة الإلغاء وشهر الوصول: 70% Train, 15% Val, 15% Test
    strat_key = df_valid["IsCanceled"].astype(str) + "_" + df_valid["ArrivalDateMonth"].astype(str)

    df_train, df_temp = train_test_split(
        df_valid, test_size=0.30, stratify=strat_key, random_state=seed
    )
    strat_temp = df_temp["IsCanceled"].astype(str) + "_" + df_temp["ArrivalDateMonth"].astype(str)
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, stratify=strat_temp, random_state=seed
    )

    df_train.to_parquet(os.path.join(splits_dir, "train.parquet"), index=False)
    df_val.to_parquet(os.path.join(splits_dir, "val.parquet"), index=False)
    df_test.to_parquet(os.path.join(splits_dir, "test_locked.parquet"), index=False)

    return df_train, df_val, df_test


def load_splits(splits_dir: str = "data/splits", allow_test_unlock: bool = False):
    df_train = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(splits_dir, "val.parquet"))
    if allow_test_unlock:
        df_test = pd.read_parquet(os.path.join(splits_dir, "test_locked.parquet"))
        return df_train, df_val, df_test
    return df_train, df_val, None
