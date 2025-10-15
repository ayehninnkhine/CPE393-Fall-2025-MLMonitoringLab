# MLflow + Evidently AI: Student Lab
**Objective:** Train and track a classification model with MLflow, then evaluate and monitor data/model drift with Evidently. Log Evidently reports as MLflow artifacts.

## What you'll learn
- Set up an experiment in **MLflow** and log parameters, metrics, plots, and the model.
- Use **Evidently** to generate **Data Drift**, **Target Drift**, and **Classification Performance** reports.
- Log Evidently **HTML** and **JSON** to MLflow runs.
- Interpret drift and performance results and write a short analysis.

## Repo structure
```
mlflow_evidently_lab/
  ├─ data/
  │   ├─ train.csv
  │   └─ test.csv
  ├─ lab_student.py
  ├─ lab_solution.py
  ├─ requirements.txt
  ├─ .gitignore
  └─ README.md
```

## Setup
```bash
# 1) Create and activate a virtual env (example with venv)
python3 -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt
```

## Part 1 — Run the student script
```bash
python lab_student.py
```
This will:
- Load `data/train.csv` (reference) and `data/test.csv` (current).
- Train a classifier.
- Log metrics, params, plots, and model with MLflow.
- Generate Evidently reports and log them as artifacts.

## Part 2 — Explore MLflow UI
```bash
mlflow server --host 127.0.0.1 --port 8080 #port number may depend on your installation
```
Then open **http://127.0.0.1:8080** and compare runs, parameters, metrics, and artifacts.
- Find the **Evidently HTML report** in the artifacts.
- Download it and inspect the drift/performance sections.

## Part 3 — Deliverables
Submit a short PDF (max 2 pages) that includes:
1. A brief description of the model, features, and metrics you tracked.
2. 1–2 screenshots from MLflow showing your metrics/plots.
3. 1 screenshot from the Evidently HTML report highlighting any **data** or **target** drift.
4. Your interpretation: What changed between reference and current data? How might that impact model performance? What would you do next in production?

## (Optional) Extensions
- Add another model (e.g., RandomForest) and compare in MLflow.
- Schedule a periodic job that regenerates the Evidently report daily/weekly.
- Log a confusion matrix image and an ROC curve to artifacts.
