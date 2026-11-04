import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


def generate_eda_report_and_plots(df: pd.DataFrame, fig_dir: str = "outputs/figures/eda", table_dir: str = "outputs/tables"):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    total_guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    total_nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]

    anomalies = {
        "Total_Rows": len(df),
        "Exact_Duplicate_Rows": int(df.duplicated().sum()),
        "Zero_Total_Guests_Rows": int((total_guests == 0).sum()),
        "Zero_Total_Nights_Rows": int((total_nights == 0).sum()),
        "NonPositive_ADR_Rows": int((df["ADR"] <= 0).sum()),
        "Extreme_ADR_GT_1000_Rows": int((df["ADR"] > 1000).sum()),
        "Children_NA_Count": int(df["Children"].isna().sum()),
        "Country_NULL_Count": int((df["Country"] == "NULL").sum()),
        "Agent_NULL_Count": int((df["Agent"] == "NULL").sum()),
        "Company_NULL_Count": int((df["Company"] == "NULL").sum()),
    }
    pd.DataFrame([anomalies]).to_csv(os.path.join(table_dir, "eda_anomalies_summary.csv"), index=False)
    df.describe(include="all").T.to_csv(os.path.join(table_dir, "eda_summary.csv"))

    # 1. توزيع ADR قبل وبعد التحويل اللوغاريتمي
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    valid_adr = df[(df["ADR"] > 0) & (df["ADR"] < 1000)]["ADR"]
    sns.histplot(valid_adr, bins=50, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title("ADR Distribution (0 < ADR < 1000)")
    sns.histplot(np.log1p(valid_adr), bins=50, kde=True, ax=axes[1], color="#2ca02c")
    axes[1].set_title("Log1p(ADR) Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_adr_distribution.png"), dpi=150)
    plt.close()

    # 2. توزيع IsCanceled ونسبة الإلغاء حسب نوع الإيداع DepositType
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=df, x="IsCanceled", ax=axes[0], palette="Blues")
    axes[0].set_title("IsCanceled Class Balance (H2 City Hotel)")
    sns.barplot(data=df, x="DepositType", y="IsCanceled", ax=axes[1], palette="Set2")
    axes[1].set_title("Cancellation Rate by DepositType")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_cancellation_and_deposit.png"), dpi=150)
    plt.close()

    # 3. ADR حسب الشهور ونوع الغرفة المحجوزة
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_clean_plot = df[(df["ADR"] > 0) & (df["ADR"] < 600)]
    sns.boxplot(data=df_clean_plot, x="ArrivalDateMonth", y="ADR", order=month_order, ax=axes[0], showfliers=False)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("ADR Seasonality by Arrival Month")
    sns.boxplot(data=df_clean_plot, x="ReservedRoomType", y="ADR", ax=axes[1], showfliers=False)
    axes[1].set_title("ADR by ReservedRoomType")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_adr_by_month_and_room.png"), dpi=150)
    plt.close()

    # 4. مصفوفة الارتباط الرقمية
    num_cols = [
        "IsCanceled", "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "IsRepeatedGuest", "PreviousCancellations",
        "PreviousBookingsNotCanceled", "BookingChanges", "DaysInWaitingList",
        "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
    ]
    plt.figure(figsize=(11, 9))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix of Numerical Variables (H2 City Hotel)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "04_correlation_heatmap.png"), dpi=150)
    plt.close()

    return anomalies


def run_m1_t2_hygiene_experiment(df_train: pd.DataFrame, seed: int = 42):
    num_features = [
        "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "PreviousCancellations",
        "PreviousBookingsNotCanceled", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
    ]

    X_raw = df_train[num_features].copy()
    y_reg_raw = df_train["ADR"].values
    y_clf_raw = df_train["IsCanceled"].values

    guests = df_train["Adults"] + df_train["Children"].fillna(0) + df_train["Babies"]
    clean_mask = (guests > 0) & (df_train["ADR"] > 0)
    df_clean = df_train.loc[clean_mask].copy()
    adr_cap = np.quantile(df_clean["ADR"], 0.999)
    df_clean["ADR_clipped"] = np.clip(df_clean["ADR"], a_min=0.01, a_max=adr_cap)

    X_clean = df_clean[num_features].copy()
    X_clean["has_agent"] = (df_clean["Agent"] != "NULL").astype(int)
    X_clean["has_company"] = (df_clean["Company"] != "NULL").astype(int)
    y_reg_clean = df_clean["ADR_clipped"].values
    y_clf_clean = df_clean["IsCanceled"].values

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    b1 = Pipeline([("imp", SimpleImputer(strategy="mean")), ("reg", LinearRegression())])
    b1_rmse = -cross_val_score(b1, X_raw, y_reg_raw, scoring="neg_root_mean_squared_error", cv=kf).mean()

    o1 = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("reg", HuberRegressor(max_iter=500))])
    o1_rmse = -cross_val_score(o1, X_clean, y_reg_clean, scoring="neg_root_mean_squared_error", cv=kf).mean()

    results.append({"Task_ID": "M1-T2", "Model": "B1_LinearReg_Raw_Unfiltered", "Role": "Baseline", "Metric": "CV_RMSE", "Score": round(float(b1_rmse), 4)})
    results.append({"Task_ID": "M1-T2", "Model": "O1_HuberReg_Filtered_Winsorized", "Role": "Optimized", "Metric": "CV_RMSE", "Score": round(float(o1_rmse), 4)})

    b2 = Pipeline([("imp", SimpleImputer(strategy="median")), ("nb", GaussianNB())])
    b2_auc = cross_val_score(b2, X_raw, y_clf_raw, scoring="roc_auc", cv=skf).mean()

    o2 = Pipeline([("imp", SimpleImputer(strategy="median")), ("power", PowerTransformer(method="yeo-johnson")), ("nb", GaussianNB(var_smoothing=1e-2))])
    o2_auc = cross_val_score(o2, X_clean, y_clf_clean, scoring="roc_auc", cv=skf).mean()

    results.append({"Task_ID": "M1-T2", "Model": "B2_GaussianNB_Raw_Skewed", "Role": "Baseline", "Metric": "CV_ROC_AUC", "Score": round(float(b2_auc), 4)})
    results.append({"Task_ID": "M1-T2", "Model": "O2_GaussianNB_YeoJohnson_Indicators", "Role": "Optimized", "Metric": "CV_ROC_AUC", "Score": round(float(o2_auc), 4)})

    X_raw_imp = MinMaxScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(X_raw.iloc[:10000]))
    km_b3 = KMeans(n_clusters=4, init="random", n_init=1, random_state=seed).fit(X_raw_imp)
    b3_sil = silhouette_score(X_raw_imp, km_b3.labels_, sample_size=5000, random_state=seed)

    X_cln_sub = np.log1p(np.maximum(0, SimpleImputer(strategy="median").fit_transform(X_clean[num_features].iloc[:10000])))
    X_cln_rob = RobustScaler().fit_transform(X_cln_sub)
    km_o3 = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=seed).fit(X_cln_rob)
    o3_sil = silhouette_score(X_cln_rob, km_o3.labels_, sample_size=5000, random_state=seed)

    results.append({"Task_ID": "M1-T2", "Model": "B3_KMeans_Raw_MinMaxScaler", "Role": "Baseline", "Metric": "Silhouette_Score", "Score": round(float(b3_sil), 4)})
    results.append({"Task_ID": "M1-T2", "Model": "O3_KMeans_Log1p_RobustScaler", "Role": "Optimized", "Metric": "Silhouette_Score", "Score": round(float(o3_sil), 4)})

    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/tables/m1_t2_hygiene_experiment.csv", index=False)
    return res_df
