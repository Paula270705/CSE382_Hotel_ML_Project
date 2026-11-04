import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import root_mean_squared_error, roc_auc_score


class ArtifactManager:
    def __init__(self, base_dir: str = "artifacts"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_artifact(self, obj, task: str, name: str) -> str:
        task_dir = os.path.join(self.base_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        filepath = os.path.join(task_dir, f"{name}.joblib")
        joblib.dump(obj, filepath)
        return filepath

    def load_artifact(self, task: str, name: str):
        filepath = os.path.join(self.base_dir, task, f"{name}.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Artifact not found: {filepath}")
        return joblib.load(filepath)


def run_m1_t1_harness_experiment(X_train_num, y_reg_train, y_clf_train, X_val_num, y_reg_val, y_clf_val, seed=42):
    manager = ArtifactManager()
    results = []

    models = {
        "B1_DummyRegressor": ("regression", DummyRegressor(strategy="mean")),
        "B2_DummyClassifier": ("classification", DummyClassifier(strategy="stratified", random_state=seed)),
        "B3_MiniBatchKMeans_Random": ("clustering", MiniBatchKMeans(n_clusters=3, init="random", n_init=1, random_state=seed)),
        "O1_Ridge_LogTarget_Pipeline": (
            "regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("reg", TransformedTargetRegressor(
                    regressor=Ridge(alpha=1.0, random_state=seed),
                    func=np.log1p,
                    inverse_func=np.expm1
                ))
            ])
        ),
        "O2_Calibrated_LogReg_Pipeline": (
            "classification",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(LogisticRegression(C=0.5, max_iter=500, random_state=seed), cv=3))
            ])
        ),
        "O3_MiniBatchKMeans_KMeansPP_Pipeline": (
            "clustering",
            Pipeline([
                ("scaler", StandardScaler()),
                ("cluster", MiniBatchKMeans(n_clusters=3, init="k-means++", batch_size=2048, n_init=10, random_state=seed))
            ])
        ),
    }

    for name, (task, model) in models.items():
        if task == "regression":
            pos_mask_tr = y_reg_train > 0
            pos_mask_val = y_reg_val > 0
            model.fit(X_train_num[pos_mask_tr], y_reg_train[pos_mask_tr])
            mem_pred = model.predict(X_val_num[pos_mask_val])
            metric_val = root_mean_squared_error(y_reg_val[pos_mask_val], mem_pred)
            metric_name = "Val_RMSE"
            X_eval = X_val_num[pos_mask_val]
        elif task == "classification":
            model.fit(X_train_num, y_clf_train)
            mem_pred = model.predict_proba(X_val_num)[:, 1]
            metric_val = roc_auc_score(y_clf_val, mem_pred)
            metric_name = "Val_ROC_AUC"
            X_eval = X_val_num
        else:
            model.fit(X_train_num)
            mem_pred = model.predict(X_val_num)
            cluster_step = model.named_steps["cluster"] if isinstance(model, Pipeline) else model
            metric_val = float(cluster_step.inertia_)
            metric_name = "Train_Inertia"
            X_eval = X_val_num

        manager.save_artifact(model, "m1_t1_verify", name)
        t0 = time.perf_counter()
        reloaded = manager.load_artifact("m1_t1_verify", name)
        load_ms = (time.perf_counter() - t0) * 1000.0

        disk_pred = reloaded.predict_proba(X_eval)[:, 1] if task == "classification" else reloaded.predict(X_eval)
        parity_err = float(np.max(np.abs(mem_pred - disk_pred)))

        results.append({
            "Task_ID": "M1-T1",
            "Model_Name": name,
            "Role": "Baseline" if name.startswith("B") else "Optimized",
            "Target_Task": task,
            "Primary_Metric": metric_name,
            "Metric_Value": round(metric_val, 4),
            "Load_Latency_ms": round(load_ms, 2),
            "Max_Serialization_Parity_Error": parity_err
        })

    df_res = pd.DataFrame(results)
    os.makedirs("outputs/tables", exist_ok=True)
    df_res.to_csv("outputs/tables/m1_t1_serialization_benchmark.csv", index=False)
    return df_res
