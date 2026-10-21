"""Общие пути артефактов и параметры пайплайна.

Все параметры можно переопределить переменными окружения в docker-compose.yaml.
"""

import os
from pathlib import Path

TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/opt/airflow/training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/airflow/models"))
DQ_REPORTS_DIR = Path(os.getenv("DQ_REPORTS_DIR", "/opt/airflow/dq_reports"))
FRAUD_SRC = os.getenv("FRAUD_SRC", "/opt/airflow/fraud_src")

CURRENT_DATASET = TRAINING_DIR / "current.parquet"
PENDING_WINDOW = TRAINING_DIR / "pending_window.parquet"
BATCHES_DIR = TRAINING_DIR / "batches"
SPLITS_DIR = TRAINING_DIR / "splits"
SUPPLY_BATCH = TRAINING_DIR / "supply_batch.parquet"
DQ_WORK_DIR = TRAINING_DIR / "dq"
REJECTED_DIR = MODELS_DIR / "rejected"

CAT_FEATURES = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]
LABEL = "target"

# --- Параметры поставки и окна (свои значения, отличаются от семинара) ---
# Размер батча, который DAG supply_labeled_batch дописывает в витрину за один запуск.
SAMPLE_SIZE = int(os.getenv("SUPPLY_SAMPLE_SIZE", "50000"))
# Глубина хранения витрины в годах относительно самого нового месяца.
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "2"))
# Число месяцев в скользящем окне обучения: (WINDOW_MONTHS - 2) train,
# 1 месяц validation для early stopping, 1 самый свежий месяц — test.
WINDOW_MONTHS = int(os.getenv("WINDOW_MONTHS", "4"))

# --- Quality gate модели ---
# Основная метрика — PR-AUC: при доле фрода < 1% она чувствительнее ROC-AUC.
# Новая модель публикуется, если PR-AUC >= PR-AUC прода + MIN_PR_AUC_GAIN
# и ROC-AUC не просел больше чем на ROC_AUC_TOLERANCE.
MIN_PR_AUC_GAIN = float(os.getenv("MIN_PR_AUC_GAIN", "0.0"))
ROC_AUC_TOLERANCE = float(os.getenv("ROC_AUC_TOLERANCE", "0.005"))
# Версия, которую fraud_detector кладёт в каталог из образа до первого обучения.
BOOTSTRAP_VERSION = "initial"
THRESHOLD = 0.5

# --- Data Quality: пороги правил (описаны в README) ---
DQ_MIN_MONTH_ROWS = int(os.getenv("DQ_MIN_MONTH_ROWS", "5000"))
DQ_MAX_OPTIONAL_NULL_SHARE = float(os.getenv("DQ_MAX_OPTIONAL_NULL_SHARE", "0.01"))
DQ_AMOUNT_MAX = float(os.getenv("DQ_AMOUNT_MAX", "30000"))
DQ_MAX_AMOUNT_OUTLIER_SHARE = float(os.getenv("DQ_MAX_AMOUNT_OUTLIER_SHARE", "0.001"))
DQ_MAX_DUPLICATE_SHARE = float(os.getenv("DQ_MAX_DUPLICATE_SHARE", "0.005"))
DQ_FRAUD_RATE_MIN = float(os.getenv("DQ_FRAUD_RATE_MIN", "0.001"))
DQ_FRAUD_RATE_MAX = float(os.getenv("DQ_FRAUD_RATE_MAX", "0.03"))
DQ_PSI_MAX = float(os.getenv("DQ_PSI_MAX", "0.2"))
DQ_ALLOWED_GENDERS = ("F", "M")
