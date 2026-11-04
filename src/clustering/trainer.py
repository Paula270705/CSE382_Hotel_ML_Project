import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.inference.artifact_manager import ArtifactManager

CLUSTER_NUM_FEATURES = [
    "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled",
    "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
]

CLUSTER_CAT_FEATURES = ["Meal", "MarketSegment", "DistributionChannel", "CustomerType", "DepositType"]


def prepare_clustering_matrices(df_train: pd.DataFrame):
    """
    يجهز مصفوفة خصائص السلوك والطلب بدون أي متغيرات نتيجة (IsCanceled / ReservationStatus).
    """
    df_c = df_train[(df_train["ADR"] > 0) & (df_train["ADR"] < 1000)].copy().reset_index(drop=True)
    df_c["Children"] = pd.to_numeric(df_c["Children"], errors="coerce").fillna(0)

    # نسخة قياسية (للـ Baselines)
    base_pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), CLUSTER_NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CLUSTER_CAT_FEATURES)
    ])
    X_base = base_pre.fit_transform(df_c)

    # نسخة محسنة بمعالجة الالتواء log1p + RobustScaler + PCA لتقليل الضوضاء (للـ Optimized)
    df_opt = df_c.copy()
    for skewed in ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "ADR", "PreviousCancellations"]:
        df_opt[skewed] = np.log1p(np.maximum(0, df_opt[skewed]))

    opt_pre = Pipeline([
        ("col", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())]), CLUSTER_NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02, sparse_output=False), CLUSTER_CAT_FEATURES)
        ])),
        ("pca_denoise", PCA(n_components=0.90, random_state=42))
    ])
    X_opt = opt_pre.fit_transform(df_opt)

    return df_c, X_base, X_opt, opt_pre


def run_clustering_experiments(df_train: pd.DataFrame, seed: int = 42):
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures/clustering", exist_ok=True)
    manager = ArtifactManager()

    df_c, X_base, X_opt, opt_pre = prepare_clustering_matrices(df_train)

    # عينة طبقية ممثلة N=10,000 لخوارزمية Agglomerative Clustering كما تنص الوثيقة (صفحة 7)
    rng = np.random.RandomState(seed)
    sub_idx = rng.choice(len(df_c), size=min(10000, len(df_c)), replace=False)

    # البحث المنهجي عن أفضل عدد مجموعات k* بدون استخدام أي Labels
    k_candidates = [2, 3, 4, 5, 6]
    k_sweep_rows = []
    best_k = 4
    best_sil = -1.0

    for k in k_candidates:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(X_opt)
        sil = silhouette_score(X_opt, km.labels_, sample_size=10000, random_state=seed)
        dbi = davies_bouldin_score(X_opt, km.labels_)
        k_sweep_rows.append({"k": k, "Inertia": float(km.inertia_), "Silhouette": float(sil), "Davies_Bouldin": float(dbi)})
        if sil > best_sil:
            best_sil = sil
            best_k = k

    df_sweep = pd.DataFrame(k_sweep_rows)
    df_sweep.to_csv("outputs/tables/clustering_k_sweep.csv", index=False)

    # رسم منحنى Elbow و Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df_sweep["k"], df_sweep["Inertia"], "o-", color="#1f77b4")
    axes[0].set_title("K-Means Elbow Curve (Inertia vs. k)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[1].plot(df_sweep["k"], df_sweep["Silhouette"], "s-", color="#2ca02c")
    axes[1].set_title("Silhouette Score vs. k")
    axes[1].set_xlabel("Number of Clusters (k)")
    plt.tight_layout()
    plt.savefig("outputs/figures/clustering/01_kmeans_elbow_silhouette.png", dpi=150)
    plt.close()

    # مقارنة الـ 3 خوارزميات في وضع Baseline ووضع Optimized
    configs = [
        ("B1_KMeans_Baseline_k3_Random", "Baseline", "Full", KMeans(n_clusters=3, init="random", n_init=1, random_state=seed), X_base),
        ("O1_KMeans_Optimized_KMeansPP", "Optimized", "Full", KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=seed), X_opt),
        ("B2_Agglomerative_Baseline_Single", "Baseline", "Subsample_10k", AgglomerativeClustering(n_clusters=3, linkage="single"), X_base[sub_idx]),
        ("O2_Agglomerative_Optimized_Ward", "Optimized", "Subsample_10k", AgglomerativeClustering(n_clusters=best_k, linkage="ward"), X_opt[sub_idx]),
        ("B3_GMM_Baseline_Spherical", "Baseline", "Full", GaussianMixture(n_components=3, covariance_type="spherical", init_params="random", random_state=seed), X_base),
        ("O3_GMM_Optimized_FullCov", "Optimized", "Full", GaussianMixture(n_components=best_k, covariance_type="full", init_params="k-means++", reg_covar=1e-4, random_state=seed), X_opt),
    ]

    results = []
    best_labels_full = None

    for name, stage, scope, algo, X_mat in configs:
        print(f"[Task C - Clustering] Running {name} ...")
        t0 = time.perf_counter()
        labels = algo.fit_predict(X_mat)
        elapsed = time.perf_counter() - t0

        n_unique = len(np.unique(labels))
        if n_unique > 1:
            sil = silhouette_score(X_mat, labels, sample_size=min(10000, len(X_mat)), random_state=seed)
            dbi = davies_bouldin_score(X_mat, labels)
            chi = calinski_harabasz_score(X_mat, labels)
        else:
            sil, dbi, chi = -1.0, 99.0, 0.0

        if name == "O1_KMeans_Optimized_KMeansPP":
            best_labels_full = labels
            manager.save_artifact({"preprocessor": opt_pre, "model": algo}, "clustering", name)

        results.append({
            "Task": "Task C - Clustering",
            "Algorithm_Name": name,
            "Stage": stage,
            "Dataset_Scope": scope,
            "Num_Clusters": int(n_unique),
            "Silhouette_Score": round(float(sil), 4),
            "Davies_Bouldin_Index": round(float(dbi), 4),
            "Calinski_Harabasz_Index": round(float(chi), 1),
            "Runtime_Sec": round(float(elapsed), 2)
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/tables/task_c_clustering_m1_results.csv", index=False)

    # رسم 2D PCA Visualization للمجموعات المكتشفة
    pca_2d = PCA(n_components=2, random_state=seed).fit_transform(X_opt[:10000])
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_2d[:, 0], y=pca_2d[:, 1], hue=best_labels_full[:10000], palette="tab10", s=15, alpha=0.7)
    plt.title(f"2D PCA Projection of Discovered Booking Segments (Optimized K-Means, k={best_k})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig("outputs/figures/clustering/02_pca_2d_clusters.png", dpi=150)
    plt.close()

    # رسم Dendrogram للتقسيم الهرمي (Ward Linkage)
    Z = linkage(X_opt[sub_idx[:2000]], method="ward")
    plt.figure(figsize=(10, 4.8))
    dendrogram(Z, truncate_mode="lastp", p=20, show_leaf_counts=True)
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage, Truncated Top 20 Merges)")
    plt.xlabel("Cluster Size")
    plt.ylabel("Ward Distance")
    plt.tight_layout()
    plt.savefig("outputs/figures/clustering/03_hierarchical_dendrogram.png", dpi=150)
    plt.close()

    # جدول تفسير خصائص كل Cluster (Cluster Personas)
    df_profile = df_c.copy()
    df_profile["Cluster"] = best_labels_full
    profile_summary = df_profile.groupby("Cluster").agg(
        Booking_Count=("ADR", "count"),
        Mean_ADR=("ADR", "mean"),
        Mean_LeadTime=("LeadTime", "mean"),
        Mean_WeekNights=("StaysInWeekNights", "mean"),
        Mean_WeekendNights=("StaysInWeekendNights", "mean"),
        Mean_Adults=("Adults", "mean"),
        Mean_SpecialRequests=("TotalOfSpecialRequests", "mean"),
        Top_MarketSegment=("MarketSegment", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A")
    ).reset_index()
    profile_summary.to_csv("outputs/tables/cluster_profiles.csv", index=False)

    return res_df, profile_summary
