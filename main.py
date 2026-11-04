import argparse
import numpy as np
import pandas as pd
from src.data.loader import load_and_verify_h2, export_schema_mapping
from src.data.eda import generate_eda_report_and_plots, run_m1_t2_hygiene_experiment
from src.data.splitter import create_and_save_splits, load_splits
from src.preprocessing.pipelines import export_leakage_audit_matrix, run_m1_t3_pipeline_ablation_experiment, SAFE_T0_NUMERIC
from src.inference.artifact_manager import run_m1_t1_harness_experiment, ArtifactManager
from src.regression.trainer import run_regression_experiments
from src.classification.trainer import run_classification_experiments
from src.clustering.trainer import run_clustering_experiments
from src.features.dimensionality import run_m1_t7_pca_lda_pilot


def cmd_analyze():
    print("=== [Step 1 & 2] Loading H2.csv, Verifying Schema & Running EDA ===")
    df = load_and_verify_h2("data/raw/H2.csv")
    export_schema_mapping(df)
    anomalies = generate_eda_report_and_plots(df)
    print("Anomalies Summary:", anomalies)

    print("=== [Step 3] Creating Stratified Train/Val/Test Splits & Exporting Leakage Audit ===")
    df_train, df_val, _ = create_and_save_splits(df)
    export_leakage_audit_matrix()

    print("=== Running Foundational 6-Model Experiments (M1-T1, M1-T2, M1-T3, M1-T7) ===")
    X_tr_num = df_train[SAFE_T0_NUMERIC].fillna(0).values
    X_val_num = df_val[SAFE_T0_NUMERIC].fillna(0).values
    run_m1_t1_harness_experiment(
        X_tr_num, df_train["ADR"].values, df_train["IsCanceled"].values,
        X_val_num, df_val["ADR"].values, df_val["IsCanceled"].values
    )
    run_m1_t2_hygiene_experiment(df_train)
    run_m1_t3_pipeline_ablation_experiment(df_train)
    run_m1_t7_pca_lda_pilot(df_train)
    print("=== Analysis & Foundational Ablations Complete! Check outputs/tables/ and outputs/figures/ ===")


def cmd_train(task: str):
    df_train, df_val, _ = load_splits("data/splits", allow_test_unlock=False)
    if task in ["regression", "all"]:
        res_reg = run_regression_experiments(df_train, df_val)
        print("\n--- Task A Regression Results ---")
        print(res_reg[["Model_Name", "Stage", "CV_RMSE_Mean", "Val_MAE", "Val_RMSE", "Val_R2"]])
    if task in ["classification", "all"]:
        res_clf = run_classification_experiments(df_train, df_val)
        print("\n--- Task B Classification Results ---")
        print(res_clf[["Model_Name", "Stage", "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1", "Val_ROC_AUC"]])


def cmd_cluster():
    df_train, _, _ = load_splits("data/splits", allow_test_unlock=False)
    res_clust, profiles = run_clustering_experiments(df_train)
    print("\n--- Task C Clustering Results ---")
    print(res_clust)
    print("\n--- Discovered Cluster Profiles ---")
    print(profiles)


def cmd_predict(task: str):
    _, df_val, _ = load_splits("data/splits", allow_test_unlock=False)
    sample = df_val.iloc[[0]]
    manager = ArtifactManager()
    if task == "regression":
        pipe = manager.load_artifact("regression", "O3_XGBoost_Tuned")
        pred = pipe.predict(sample)[0]
        print(f"Sample Booking Predicted ADR: {pred:.2f} (Actual ADR: {sample['ADR'].iloc[0]:.2f})")
    else:
        pipe = manager.load_artifact("classification", "O3_XGBoost_Tuned_Weighted")
        prob = pipe.predict_proba(sample)[0, 1]
        print(f"Sample Booking Cancellation Probability: {prob:.3f} (Actual IsCanceled: {sample['IsCanceled'].iloc[0]})")


def main():
    parser = argparse.ArgumentParser(description="CSE382 City Hotel ML System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("analyze", help="Run EDA, splits, leakage audit, and foundational experiments")

    p_train = subparsers.add_parser("train", help="Train baseline and optimized supervised models")
    p_train.add_argument("--task", choices=["regression", "classification", "all"], default="all")
    p_train.add_argument("--model", default="all")

    p_cluster = subparsers.add_parser("cluster", help="Run baseline and optimized clustering algorithms")
    p_cluster.add_argument("--algorithm", default="all")

    p_pred = subparsers.add_parser("predict", help="Run inference using saved artifacts")
    p_pred.add_argument("--task", choices=["regression", "classification"], required=True)

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze()
    elif args.command == "train":
        cmd_train(args.task)
    elif args.command == "cluster":
        cmd_cluster()
    elif args.command == "predict":
        cmd_predict(args.task)


if __name__ == "__main__":
    main()
