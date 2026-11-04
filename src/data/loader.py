import os
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate"
]


def load_and_verify_h2(filepath: str = "data/raw/H2.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"H2 dataset not found at {filepath}")

    df = pd.read_csv(filepath, keep_default_na=False, na_values=["", "NA", "NaN"])
    df.columns = [c.strip() for c in df.columns]

    assert df.shape == (79330, 31), f"Expected shape (79330, 31), got {df.shape}"
    assert list(df.columns) == EXPECTED_COLUMNS, "Columns do not match official H2 schema!"

    df["Children"] = pd.to_numeric(df["Children"], errors="coerce")
    df["ADR"] = pd.to_numeric(df["ADR"], errors="coerce")

    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()

    return df


def export_schema_mapping(df: pd.DataFrame, out_path: str = "outputs/tables/schema_mapping.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for col in df.columns:
        null_literal_cnt = int((df[col].astype(str) == "NULL").sum())
        na_cnt = int(df[col].isna().sum())
        rows.append({
            "Source_Label": col,
            "Code_Name": col,
            "Dtype": str(df[col].dtype),
            "N_Unique": int(df[col].nunique(dropna=True)),
            "NA_Count": na_cnt,
            "String_NULL_Count": null_literal_cnt,
            "Sample_Values": str(df[col].dropna().unique()[:4].tolist())
        })
    schema_df = pd.DataFrame(rows)
    schema_df.to_csv(out_path, index=False)
    return schema_df
