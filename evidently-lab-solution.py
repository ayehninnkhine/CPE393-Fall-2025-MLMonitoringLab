import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    ClassificationPreset,
)
from evidently import ColumnMapping

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(TRACKING_URI)
EXP_NAME = "mlflow-evidently-lab"
mlflow.set_experiment(EXP_NAME)

def load_data():
    ref = pd.read_csv(DATA_DIR / "train.csv")
    cur = pd.read_csv(DATA_DIR / "test.csv")
    return ref, cur

def make_models():
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=0,
        n_jobs=-1
    )
    return {"LogReg": lr, "RandomForest": rf}

def log_plots(y_true, y_pred, y_proba, tag_prefix=""):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{tag_prefix} Confusion Matrix".strip())
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    cm_path = REPORTS_DIR / f"{tag_prefix.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.tight_layout(); plt.savefig(cm_path, dpi=150); plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    # ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig2 = plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"{tag_prefix} ROC Curve".strip())
        plt.xlabel("FPR"); plt.ylabel("TPR")
        roc_path = REPORTS_DIR / f"{tag_prefix.lower().replace(' ', '_')}_roc_curve.png"
        plt.tight_layout(); plt.savefig(roc_path, dpi=150); plt.close(fig2)
        mlflow.log_artifact(str(roc_path))

def run_model(name, model, ref_df, cur_df):
    target = "target"
    X_train = ref_df.drop(columns=[target])
    y_train = ref_df[target].astype(int)
    X_test  = cur_df.drop(columns=[target])
    y_test  = cur_df[target].astype(int)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1", float(f1))
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", float(roc_auc))

        log_plots(y_test, y_pred, y_proba, tag_prefix=name)

        input_example = X_train.head(5)
        try:
            signature = infer_signature(X_train, model.predict(X_train))
        except Exception:
            signature = None

        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )
        except Exception:
            pass

        ref_df_copy = ref_df.copy()
        cur_df_copy = cur_df.copy()

        ref_df_copy["prediction"] = model.predict(X_train).astype(int)
        if hasattr(model, "predict_proba"):
            try:
                ref_p = model.predict_proba(X_train)[:, 1]
                ref_df_copy["proba_1"] = ref_p
                ref_df_copy["proba_0"] = 1.0 - ref_p
            except Exception:
                pass

        cur_df_copy["prediction"] = y_pred.astype(int)
        if y_proba is not None:
            cur_df_copy["proba_1"] = y_proba
            cur_df_copy["proba_0"] = 1.0 - y_proba

        mapping = ColumnMapping()
        mapping.target = target
        mapping.prediction = "prediction"
        mapping.prediction_probas = ["proba_0", "proba_1"]

        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            ClassificationPreset(),
        ])
        report.run(reference_data=ref_df_copy, current_data=cur_df_copy, column_mapping=mapping)

        html_path = REPORTS_DIR / f"{name}_evidently_report.html"
        json_path = REPORTS_DIR / f"{name}_evidently_report.json"
        report.save_html(str(html_path))
        report.save_json(str(json_path))
        mlflow.log_artifact(str(html_path))
        mlflow.log_artifact(str(json_path))

        msg = f"[{name}] acc={acc:.3f} f1={f1:.3f}"
        if 'roc_auc' in locals():
            msg += f" roc_auc={roc_auc:.3f}"
        print(msg)

# ---- main ----
if __name__ == "__main__":
    ref, cur = load_data()
    for n, m in make_models().items():
        run_model(n, m, ref, cur)
    print("Done. Launch MLflow UI to inspect runs and Evidently reports.")
