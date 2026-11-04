import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)
from src.preprocessing.pipelines import build_supervised_preprocessor
from src.inference.artifact_manager import ArtifactManager


def get_classification_models(seed: int = 42):
    """
    يرجع 3 موديلات Baseline غير مضبوطة و 3 موديلات Optimized مضبوطة لمهمة التصنيف (IsCanceled).
    """
    return {
        # --- 3 Baseline Models (Untuned) ---
        "B1_LogisticRegression_Baseline": Pipeline([
            ("pre", build_supervised_preprocessor("classification")),
            ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
        ]),
        "B2_RandomForest_Baseline": Pipeline([
            ("pre", build_supervised_preprocessor("classification")),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=seed))
        ]),
        "B3_XGBoost_Baseline": Pipeline([
            ("pre", build_supervised_preprocessor("classification")),
            ("model", XGBClassifier(
                n_estimators=100, learning_rate=0.3, max_depth=6,
                eval_metric="logloss", n_jobs=-1, random_state=seed
            ))
        ]),

        # --- 3 Optimized Models (Class-Balanced & Hyperparameter-Tuned) ---
        "O1_LogisticRegression_ElasticNet_Balanced": Pipeline([
            ("pre", build_supervised_preprocessor("classification")),
            ("model", LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
                class_weight="balanced", max_iter=1000, random_state=seed
            ))
        ]),
        "O2_RandomForest_Tuned_Balanced": Pipeline([
            ("pre", build_supervised_preprocessor("classification")),
            ("model", RandomForestClassifier(
                n_estimators=250, max_depth=22, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed
            ))
        ]),
        "O3_XGBoost_Tuned_Weighted": Pipeline([
            ("pre", build_supervised_preprocessor("classification")),
            ("model", XGBClassifier(
                n_estimators=400, learning_rate=0.06, max_depth=8,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=1.35, reg_lambda=5.0,
                eval_metric="logloss", n_jobs=-1, random_state=seed
            ))
        ]),
    }


def run_classification_experiments(df_train: pd.DataFrame, df_val: pd.DataFrame, seed: int = 42):
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures/classification", exist_ok=True)
    manager = ArtifactManager()

    y_train = df_train["IsCanceled"].values
    y_val = df_val["IsCanceled"].values

    models = get_classification_models(seed=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []

    plt.figure(figsize=(8, 6))
    best_f1 = -1.0
    best_cm = None
    best_name = ""

    for name, pipe in models.items():
        print(f"[Task B - Classification] Running {name} ...")
        t0 = time.perf_counter()

        cv_scores = cross_validate(
            pipe, df_train, y_train, cv=skf,
            scoring={
                "acc": "accuracy",
                "f1": "f1",
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision"
            },
            n_jobs=1
        )

        pipe.fit(df_train, y_train)
        train_time = time.perf_counter() - t0

        y_prob = pipe.predict_proba(df_val)[:, 1]
        # معايرة عتبة القرار (Threshold Calibration) للموديلات المحسنة لتعظيم F1
        if name.startswith("O"):
            thresholds = np.linspace(0.35, 0.65, 31)
            f1s = [f1_score(y_val, (y_prob >= t).astype(int)) for t in thresholds]
            best_t = float(thresholds[int(np.argmax(f1s))])
        else:
            best_t = 0.50

        y_pred = (y_prob >= best_t).astype(int)
        cm = confusion_matrix(y_val, y_pred)

        val_acc = accuracy_score(y_val, y_pred)
        val_prec = precision_score(y_val, y_pred, zero_division=0)
        val_rec = recall_score(y_val, y_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_pred, zero_division=0)
        val_roc = roc_auc_score(y_val, y_prob)
        val_pr = average_precision_score(y_val, y_prob)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_cm = cm
            best_name = name

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={val_roc:.3f})")

        manager.save_artifact(pipe, "classification", name)

        rows.append({
            "Task": "Task B - Classification (IsCanceled)",
            "Model_Name": name,
            "Stage": "Baseline" if name.startswith("B") else "Optimized",
            "Decision_Threshold": round(best_t, 2),
            "CV_F1_Mean": round(float(cv_scores["test_f1"].mean()), 4),
            "CV_ROC_AUC_Mean": round(float(cv_scores["test_roc_auc"].mean()), 4),
            "Val_Accuracy": round(float(val_acc), 4),
            "Val_Precision": round(float(val_prec), 4),
            "Val_Recall": round(float(val_rec), 4),
            "Val_F1": round(float(val_f1), 4),
            "Val_ROC_AUC": round(float(val_roc), 4),
            "Val_PR_AUC": round(float(val_pr), 4),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
            "Train_Time_Sec": round(float(train_time), 2)
        })

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task B (IsCanceled): Validation ROC Curves (Baseline vs. Optimized)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/figures/classification/classification_roc_curves.png", dpi=150)
    plt.close()

    # حفظ رسم Confusion Matrix لأفضل موديل
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig("outputs/figures/classification/best_confusion_matrix.png", dpi=150)
    plt.close()

    res_df = pd.DataFrame(rows)
    res_df.to_csv("outputs/tables/task_b_classification_m1_results.csv", index=False)
    return res_df
