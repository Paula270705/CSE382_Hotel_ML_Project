import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from src.preprocessing.pipelines import build_supervised_preprocessor


def run_m1_t7_pca_lda_pilot(df_train: pd.DataFrame, seed: int = 42):
    """
    ينفذ تجربة M1-T7 لتجهيز PCA و LDA ورسم منحنى التباين المفسر (Explained Variance)
    واختبار 3 موديلات Baseline مقابل 3 موديلات Optimized جاهزة لـ Milestone 2.
    """
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    df_sub = df_train[(df_train["ADR"] > 0) & (df_train["ADR"] < 1000)].iloc[:12000].copy()
    y_reg = df_sub["ADR"].values
    y_clf = df_sub["IsCanceled"].values

    # حساب ورسم منحنى التباين التراكمي للـ PCA
    pre_clf = build_supervised_preprocessor("classification")
    X_trans = pre_clf.fit_transform(df_sub, y_clf)
    pca_full = PCA(random_state=seed).fit(X_trans)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cum_var, 0.95) + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(cum_var) + 1), cum_var, "b-", lw=2, label="Cumulative Explained Variance")
    plt.axhline(0.95, color="r", linestyle="--", label=f"95% Variance Threshold ({n_95} components)")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA Cumulative Explained Variance Curve (H2 City Hotel)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/pca_explained_variance.png", dpi=150)
    plt.close()

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []

    # Pair 1: KNN Regressor (Uncompressed vs PCA 95% Whitened + Distance Weighting)
    b1 = Pipeline([
        ("pre", build_supervised_preprocessor("regression")),
        ("knn", KNeighborsRegressor(n_neighbors=5, weights="uniform"))
    ])
    b1_rmse = -cross_val_score(b1, df_sub, y_reg, scoring="neg_root_mean_squared_error", cv=kf).mean()

    o1 = Pipeline([
        ("pre", build_supervised_preprocessor("regression", dim_reduction_step=PCA(n_components=0.95, whiten=True, random_state=seed))),
        ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance", metric="manhattan"))
    ])
    o1_rmse = -cross_val_score(o1, df_sub, y_reg, scoring="neg_root_mean_squared_error", cv=kf).mean()

    results.append({"Task_ID": "M1-T7", "Model": "B1_KNNRegressor_NoPCA_Uniform", "Role": "Baseline", "Metric": "CV_RMSE", "Score": round(float(b1_rmse), 4)})
    results.append({"Task_ID": "M1-T7", "Model": "O1_KNNRegressor_PCA95_Whitened_Distance", "Role": "Optimized", "Metric": "CV_RMSE", "Score": round(float(o1_rmse), 4)})

    # Pair 2: KNN Classifier (Naive 2-Component PCA vs Hybrid PCA 95% + SVD LDA Axis)
    b2 = Pipeline([
        ("pre", build_supervised_preprocessor("classification", dim_reduction_step=PCA(n_components=2, random_state=seed))),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="uniform"))
    ])
    b2_auc = cross_val_score(b2, df_sub, y_clf, scoring="roc_auc", cv=skf).mean()

    hybrid_dr = FeatureUnion([
        ("pca", PCA(n_components=0.95, random_state=seed)),
        ("lda", LinearDiscriminantAnalysis(n_components=1, solver="svd"))
    ])
    o2 = Pipeline([
        ("pre", build_supervised_preprocessor("classification", dim_reduction_step=hybrid_dr)),
        ("knn", KNeighborsClassifier(n_neighbors=21, weights="distance"))
    ])
    o2_auc = cross_val_score(o2, df_sub, y_clf, scoring="roc_auc", cv=skf).mean()

    results.append({"Task_ID": "M1-T7", "Model": "B2_KNNClassifier_Naive_PCA2", "Role": "Baseline", "Metric": "CV_ROC_AUC", "Score": round(float(b2_auc), 4)})
    results.append({"Task_ID": "M1-T7", "Model": "O2_KNNClassifier_PCA95_Plus_SVD_LDA", "Role": "Optimized", "Metric": "CV_ROC_AUC", "Score": round(float(o2_auc), 4)})

    # Pair 3: LogisticRegression on Unregularized SVD LDA vs Hybrid SVD LDA + Top PCA Axes
    b3 = Pipeline([
        ("pre", build_supervised_preprocessor("classification", dim_reduction_step=LinearDiscriminantAnalysis(n_components=1, solver="svd"))),
        ("clf", LogisticRegression(random_state=seed))
    ])
    b3_f1 = cross_val_score(b3, df_sub, y_clf, scoring="f1", cv=skf).mean()

    o3 = Pipeline([
        ("pre", build_supervised_preprocessor("classification", dim_reduction_step=hybrid_dr)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", random_state=seed))
    ])
    o3_f1 = cross_val_score(o3, df_sub, y_clf, scoring="f1", cv=skf).mean()

    results.append({"Task_ID": "M1-T7", "Model": "B3_LogReg_Unregularized_SVD_LDA", "Role": "Baseline", "Metric": "CV_F1", "Score": round(float(b3_f1), 4)})
    results.append({"Task_ID": "M1-T7", "Model": "O3_LogReg_SVD_LDA_PCA_Balanced", "Role": "Optimized", "Metric": "CV_F1", "Score": round(float(o3_f1), 4)})

    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/tables/m1_t7_pca_lda_pilot.csv", index=False)
    return res_df
