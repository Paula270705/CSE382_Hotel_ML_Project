import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from src.preprocessing.pipelines import build_supervised_preprocessor
from src.inference.artifact_manager import ArtifactManager


def get_regression_models(seed: int = 42):
    """
    يرجع 3 موديلات Baseline غير مضبوطة و 3 موديلات Optimized مضبوطة لمهمة الانحدار (ADR).
    """
    return {
        # --- 3 Baseline Models (Untuned) ---
        "B1_LinearRegression_Baseline": Pipeline([
            ("pre", build_supervised_preprocessor("regression")),
            ("model", LinearRegression())
        ]),
        "B2_RandomForest_Baseline": Pipeline([
            ("pre", build_supervised_preprocessor("regression")),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1, random_state=seed))
        ]),
        "B3_XGBoost_Baseline": Pipeline([
            ("pre", build_supervised_preprocessor("regression")),
            ("model", XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=6, n_jobs=-1, random_state=seed))
        ]),

        # --- 3 Optimized Models (Regularized & Hyperparameter-Tuned) ---
        "O1_ElasticNet_LogTarget_Optimized": Pipeline([
            ("pre", build_supervised_preprocessor("regression")),
            ("model", TransformedTargetRegressor(
                regressor=ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=1000, random_state=seed),
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ]),
        "O2_RandomForest_Tuned": Pipeline([
            ("pre", build_supervised_preprocessor("regression")),
            ("model", RandomForestRegressor(
                n_estimators=250, max_depth=22, min_samples_leaf=3,
                max_features=0.6, n_jobs=-1, random_state=seed
            ))
        ]),
        "O3_XGBoost_Tuned": Pipeline([
            ("pre", build_supervised_preprocessor("regression")),
            ("model", XGBRegressor(
                n_estimators=400, learning_rate=0.06, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
                n_jobs=-1, random_state=seed
            ))
        ]),
    }


def run_regression_experiments(df_train: pd.DataFrame, df_val: pd.DataFrame, seed: int = 42):
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures/regression", exist_ok=True)
    manager = ArtifactManager()

    # فلترة الحجوزات ذات ADR <= 0 أو الشواذ المستحيلة (> 1000) في تدريب وتقييم الانحدار
    tr_clean = df_train[(df_train["ADR"] > 0) & (df_train["ADR"] < 1000)].copy()
    val_clean = df_val[(df_val["ADR"] > 0) & (df_val["ADR"] < 1000)].copy()

    y_train = tr_clean["ADR"].values
    y_val = val_clean["ADR"].values

    models = get_regression_models(seed=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    val_preds = {}

    for name, pipe in models.items():
        print(f"[Task A - Regression] Running {name} ...")
        t0 = time.perf_counter()

        cv_scores = cross_validate(
            pipe, tr_clean, y_train, cv=kf,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            },
            n_jobs=1
        )

        pipe.fit(tr_clean, y_train)
        train_time = time.perf_counter() - t0
        y_pred = pipe.predict(val_clean)
        val_preds[name] = y_pred

        val_mae = mean_absolute_error(y_val, y_pred)
        val_mse = mean_squared_error(y_val, y_pred)
        val_rmse = root_mean_squared_error(y_val, y_pred)
        val_r2 = r2_score(y_val, y_pred)

        manager.save_artifact(pipe, "regression", name)

        rows.append({
            "Task": "Task A - Regression (ADR)",
            "Model_Name": name,
            "Stage": "Baseline" if name.startswith("B") else "Optimized",
            "CV_MAE_Mean": round(float(-cv_scores["test_mae"].mean()), 4),
            "CV_RMSE_Mean": round(float(-cv_scores["test_rmse"].mean()), 4),
            "CV_RMSE_Std": round(float(cv_scores["test_rmse"].std()), 4),
            "CV_R2_Mean": round(float(cv_scores["test_r2"].mean()), 4),
            "Val_MAE": round(float(val_mae), 4),
            "Val_MSE": round(float(val_mse), 4),
            "Val_RMSE": round(float(val_rmse), 4),
            "Val_R2": round(float(val_r2), 4),
            "Train_Time_Sec": round(float(train_time), 2)
        })

    res_df = pd.DataFrame(rows)
    res_df.to_csv("outputs/tables/task_a_regression_m1_results.csv", index=False)

    # رسم مقارنة RMSE بين الـ Baseline والـ Optimized
    plt.figure(figsize=(10, 5))
    sns.barplot(data=res_df, x="Model_Name", y="Val_RMSE", hue="Stage", palette="viridis")
    plt.xticks(rotation=30, ha="right")
    plt.title("Task A (ADR Regression): Validation RMSE — Baseline vs. Optimized")
    plt.tight_layout()
    plt.savefig("outputs/figures/regression/regression_rmse_comparison.png", dpi=150)
    plt.close()

    # رسم Predicted vs Actual لأفضل موديل
    best_model_name = res_df.sort_values("Val_RMSE").iloc[0]["Model_Name"]
    best_pred = val_preds[best_model_name]
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y_val[:2000], best_pred[:2000], alpha=0.3, s=12, color="#1f77b4")
    plt.plot([0, 350], [0, 350], "r--", lw=2)
    plt.xlim(0, 350)
    plt.ylim(0, 350)
    plt.xlabel("Actual ADR")
    plt.ylabel("Predicted ADR")
    plt.title(f"Predicted vs. Actual ADR ({best_model_name})")
    plt.tight_layout()
    plt.savefig("outputs/figures/regression/best_regression_pred_vs_actual.png", dpi=150)
    plt.close()

    return res_df
