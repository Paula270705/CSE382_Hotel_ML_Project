import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

# القوائم المعتمدة الخالية من الـ Data Leakage عند نقطة إنشاء الحجز (t0)
SAFE_T0_NUMERIC = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
]

SAFE_T0_LOW_CARD_CAT = [
    "ArrivalDateMonth", "Meal", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType"
]

SAFE_T0_HIGH_CARD_CAT = ["Country", "Agent", "Company"]


class BookingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    يشتق خصائص دومين الفنادق بطريقة آمنة وبدون أي تسريب من بيانات الاختبار.
    """
    def __init__(self, include_adr: bool = False):
        self.include_adr = include_adr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        children = pd.to_numeric(df["Children"], errors="coerce").fillna(0)
        df["Children"] = children
        df["total_nights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
        df["total_guests"] = df["Adults"] + children + df["Babies"]
        df["weekend_ratio"] = df["StaysInWeekendNights"] / (df["total_nights"] + 1.0)
        df["is_family"] = ((children + df["Babies"]) > 0).astype(int)
        df["prev_cancel_ratio"] = df["PreviousCancellations"] / (
            df["PreviousCancellations"] + df["PreviousBookingsNotCanceled"] + 1.0
        )
        df["has_agent"] = (df["Agent"].astype(str) != "NULL").astype(int)
        df["has_company"] = (df["Company"].astype(str) != "NULL").astype(int)

        month_num = df["ArrivalDateMonth"].map(MONTH_MAP).fillna(1)
        df["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
        return df


def build_supervised_preprocessor(task: str = "regression", dim_reduction_step=None):
    """
    يبني ColumnTransformer قياسي لمهام Regression أو Classification مع دعم إضافة PCA/LDA لاحقاً.
    """
    include_adr = (task == "classification")
    num_cols = list(SAFE_T0_NUMERIC) + (["ADR"] if include_adr else []) + [
        "total_nights", "total_guests", "weekend_ratio", "is_family",
        "prev_cancel_ratio", "has_agent", "has_company", "month_sin", "month_cos"
    ]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False))
    ])

    target_type = "continuous" if task == "regression" else "binary"
    high_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="NULL")),
        ("target_enc", TargetEncoder(cv=5, smooth="auto", target_type=target_type, random_state=42))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("low_cat", low_cat_pipe, SAFE_T0_LOW_CARD_CAT),
            ("high_cat", high_cat_pipe, SAFE_T0_HIGH_CARD_CAT)
        ],
        remainder="drop"
    )

    steps = [
        ("feat_eng", BookingFeatureEngineer(include_adr=include_adr)),
        ("col_trans", col_transformer)
    ]
    if dim_reduction_step is not None:
        steps.append(("dim_red", dim_reduction_step))

    return Pipeline(steps)


def export_leakage_audit_matrix(out_path: str = "outputs/tables/leakage_audit_matrix.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    audit_rows = [
        ("IsCanceled", "Excluded (Post-booking outcome)", "Target Variable", "Excluded (Supervised outcome)", "Post-booking outcome", "Direct Target / Outcome Leak"),
        ("ADR", "Target Variable", "Included (Valid t0 pricing)", "Included (Behavioral price tier)", "At booking creation (t0)", "Target in Task A / Safe in B & C"),
        ("ReservationStatus", "Excluded (Direct leak)", "Excluded (100% Direct leak of IsCanceled)", "Excluded (Outcome label)", "Final reservation state", "Direct Target Leak"),
        ("ReservationStatusDate", "Excluded (Direct leak)", "Excluded (Timestamp of cancellation/check-out)", "Excluded (Outcome timestamp)", "Final reservation state", "Direct Temporal Leak"),
        ("AssignedRoomType", "Excluded (Assigned at check-in)", "Excluded (Indirect check-in reassignment leak)", "Excluded (Operational mutation)", "Check-in / Operations", "Indirect Lifecycle Leak"),
        ("BookingChanges", "Excluded (Accumulates post-t0)", "Excluded in strict t0 (Lifecycle mutation)", "Excluded (Operational count)", "Post-booking amendments", "Indirect Lifecycle Leak"),
        ("DaysInWaitingList", "Excluded (Resolved post-wait)", "Excluded in strict t0 (Lifecycle wait)", "Excluded (Operational queue)", "Post-booking confirmation", "Indirect Lifecycle Leak"),
        ("LeadTime", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("ArrivalDateYear", "Included", "Included", "Excluded (Calendar ID)", "At booking creation (t0)", "Safe at t0"),
        ("ArrivalDateMonth", "Included", "Included", "Included (Seasonality)", "At booking creation (t0)", "Safe at t0"),
        ("ArrivalDateWeekNumber", "Included", "Included", "Excluded (Redundant with Month)", "At booking creation (t0)", "Safe at t0"),
        ("ArrivalDateDayOfMonth", "Included", "Included", "Excluded (Noise in clustering)", "At booking creation (t0)", "Safe at t0"),
        ("StaysInWeekendNights", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("StaysInWeekNights", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("Adults", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("Children", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("Babies", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("Meal", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("Country", "Included (OOF Target Encoded)", "Included (OOF Target Encoded)", "Excluded (High cardinality)", "At booking creation (t0)", "Safe at t0 (Requires regularization)"),
        ("MarketSegment", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("DistributionChannel", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("IsRepeatedGuest", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("PreviousCancellations", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("PreviousBookingsNotCanceled", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("ReservedRoomType", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("DepositType", "Included", "Included (Audited high-signal contract)", "Included", "At booking creation (t0)", "Safe at t0"),
        ("Agent", "Included (OOF Target + HasAgent)", "Included (OOF Target + HasAgent)", "Replaced by HasAgent indicator", "At booking creation (t0)", "Safe at t0 (ID requires encoding)"),
        ("Company", "Included (OOF Target + HasCompany)", "Included (OOF Target + HasCompany)", "Replaced by HasCompany indicator", "At booking creation (t0)", "Safe at t0 (ID requires encoding)"),
        ("CustomerType", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("RequiredCarParkingSpaces", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
        ("TotalOfSpecialRequests", "Included", "Included", "Included", "At booking creation (t0)", "Safe at t0"),
    ]
    df_audit = pd.DataFrame(audit_rows, columns=[
        "Source_Variable", "Task_A_Regression_Role", "Task_B_Classification_Role",
        "Task_C_Clustering_Role", "Lifecycle_Timing", "Leakage_Classification"
    ])
    df_audit.to_csv(out_path, index=False)
    return df_audit


def run_m1_t3_pipeline_ablation_experiment(df_train: pd.DataFrame, seed: int = 42):
    """
    ينفذ تجربة M1-T3 (3 موديلات Baseline مقابل 3 موديلات Optimized) لإثبات أثر التكويد الصحيح
    وإثبات خطورة الـ Data Leakage عملياً بالتجربة للأستاذ والمعيد.
    """
    df_sub = df_train[(df_train["ADR"] > 0) & (df_train["ADR"] < 1000)].iloc[:15000].copy()
    y_reg = df_sub["ADR"].values
    y_clf = df_sub["IsCanceled"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []

    # Pair 1: SGDRegressor Naive OHE vs OOF TargetEncoder + Engineered Features
    naive_pre_reg = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), SAFE_T0_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SAFE_T0_LOW_CARD_CAT + SAFE_T0_HIGH_CARD_CAT)
    ])
    b1 = Pipeline([("pre", naive_pre_reg), ("reg", SGDRegressor(max_iter=500, random_state=seed))])
    b1_rmse = -cross_val_score(b1, df_sub, y_reg, scoring="neg_root_mean_squared_error", cv=kf).mean()

    o1 = Pipeline([("pre", build_supervised_preprocessor("regression")), ("reg", SGDRegressor(max_iter=500, random_state=seed))])
    o1_rmse = -cross_val_score(o1, df_sub, y_reg, scoring="neg_root_mean_squared_error", cv=kf).mean()

    results.append({"Task_ID": "M1-T3", "Model": "B1_SGDRegressor_Naive_Full_OHE", "Role": "Baseline", "Metric": "CV_RMSE", "Score": round(float(b1_rmse), 4)})
    results.append({"Task_ID": "M1-T3", "Model": "O1_SGDRegressor_OOF_TargetEnc_Engineered", "Role": "Optimized", "Metric": "CV_RMSE", "Score": round(float(o1_rmse), 4)})

    # Pair 2: LogisticRegression Naive OHE vs OOF TargetEncoder + Engineered Ratios
    naive_pre_clf = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), SAFE_T0_NUMERIC + ["ADR"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SAFE_T0_LOW_CARD_CAT + SAFE_T0_HIGH_CARD_CAT)
    ])
    b2 = Pipeline([("pre", naive_pre_clf), ("clf", LogisticRegression(max_iter=500, random_state=seed))])
    b2_f1 = cross_val_score(b2, df_sub, y_clf, scoring="f1", cv=skf).mean()

    o2 = Pipeline([("pre", build_supervised_preprocessor("classification")), ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed))])
    o2_f1 = cross_val_score(o2, df_sub, y_clf, scoring="f1", cv=skf).mean()

    results.append({"Task_ID": "M1-T3", "Model": "B2_LogReg_Naive_OHE_NoFeatureEng", "Role": "Baseline", "Metric": "CV_F1", "Score": round(float(b2_f1), 4)})
    results.append({"Task_ID": "M1-T3", "Model": "O2_LogReg_OOF_TargetEnc_Engineered", "Role": "Optimized", "Metric": "CV_F1", "Score": round(float(o2_f1), 4)})

    # Pair 3: Empirical Leakage Audit Proof (DecisionTree WITH Indirect Leak vs Strictly Audited t0 Safe Set)
    leaky_pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), SAFE_T0_NUMERIC + ["ADR", "BookingChanges", "DaysInWaitingList"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SAFE_T0_LOW_CARD_CAT + ["AssignedRoomType"])
    ])
    b3_leaky = Pipeline([("pre", leaky_pre), ("dt", DecisionTreeClassifier(max_depth=10, random_state=seed))])
    b3_auc = cross_val_score(b3_leaky, df_sub, y_clf, scoring="roc_auc", cv=skf).mean()

    o3_safe = Pipeline([("pre", build_supervised_preprocessor("classification")), ("dt", DecisionTreeClassifier(max_depth=10, ccp_alpha=0.0005, class_weight="balanced", random_state=seed))])
    o3_auc = cross_val_score(o3_safe, df_sub, y_clf, scoring="roc_auc", cv=skf).mean()

    results.append({"Task_ID": "M1-T3", "Model": "B3_DecisionTree_WITH_Lifecycle_Leakage", "Role": "Baseline (Leaky Reference)", "Metric": "CV_ROC_AUC", "Score": round(float(b3_auc), 4)})
    results.append({"Task_ID": "M1-T3", "Model": "O3_DecisionTree_Strict_t0_LeakFree_Pruned", "Role": "Optimized (Leak-Free)", "Metric": "CV_ROC_AUC", "Score": round(float(o3_auc), 4)})

    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/tables/m1_t3_pipeline_ablation.csv", index=False)
    return res_df
