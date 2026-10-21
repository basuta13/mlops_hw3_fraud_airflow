"""DAG скользящего переобучения CatBoost с quality gate.

Файл только связывает задачи. Логика каждой задачи лежит в pipeline.retrain.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.utils.trigger_rule import TriggerRule

from pipeline.retrain.advance_training_window import advance_training_window
from pipeline.retrain.extract_month_batch import extract_month_batch
from pipeline.retrain.fit_catboost import fit_catboost
from pipeline.retrain.preprocess_dataset import preprocess_dataset
from pipeline.retrain.publish_model import publish_model, reject_model
from pipeline.retrain.quality_gate import (
    PUBLISH_TASK_ID,
    REJECT_TASK_ID,
    choose_deployment,
    evaluate_prod_model,
)
from pipeline.retrain.slide_training_window import slide_training_window
from pipeline.retrain.time_train_test_split import time_train_test_split


@dag(
    dag_id="retrain_catboost",
    description="Раз в 5 минут сдвигает окно, переобучает CatBoost и выкатывает, если модель не хуже прода",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=20),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "catboost"],
    doc_md="""
### Переобучение my_catboost с quality gate

1. `extract_month_batch` — следующий месяц из витрины по указателю `training_state`.
2. `slide_training_window` — окно из `WINDOW_MONTHS` месяцев, заканчивающееся новым месяцем.
3. `time_train_test_split` — train: ранние месяцы, valid: предпоследний, test: свежий.
4. `preprocess_dataset` — очистка train/valid + `run_preproc` из `fraud_detector/src`.
5. `fit_catboost` — кандидат, early stopping по valid.
6. `evaluate_prod_model` — прод-модель из `latest.txt` на том же test-месяце.
7. `choose_deployment` — ветка: `publish_model` (кандидат не хуже) или `reject_model`.
8. `advance_training_window` — указатель месяца сдвигается только после сохранения модели.
    """,
)
def retrain_catboost():
    """Связывает задачи переобучения в граф с ветвлением по качеству модели."""
    extracted = task(extract_month_batch, task_id="extract_month_batch")()
    slid = task(slide_training_window, task_id="slide_training_window")(extracted)
    split = task(time_train_test_split, task_id="time_train_test_split")(slid)
    processed = task(preprocess_dataset, task_id="preprocess_dataset")(split)
    trained = task(fit_catboost, task_id="fit_catboost")(processed)
    evaluation = task(evaluate_prod_model, task_id="evaluate_prod_model")(trained)

    branch = task.branch(choose_deployment, task_id="choose_deployment")(evaluation)
    published = task(publish_model, task_id=PUBLISH_TASK_ID)(evaluation)
    rejected = task(reject_model, task_id=REJECT_TASK_ID)(evaluation)
    advanced = task(
        advance_training_window,
        task_id="advance_training_window",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )(evaluation)

    branch >> [published, rejected]
    [published, rejected] >> advanced


retrain_catboost()
