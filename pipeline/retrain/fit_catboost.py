"""Обучение CatBoost-кандидата на новом окне."""

import logging
import os
from datetime import datetime, timezone

import pandas as pd

from pipeline.retrain.metrics import classification_metrics, split_xy
from pipeline.settings import CAT_FEATURES, MODELS_DIR

logger = logging.getLogger(__name__)

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.08,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "auto_class_weights": "SqrtBalanced",
    "early_stopping_rounds": 30,
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": 50,
}


def fit_catboost(preproc_info: dict) -> dict:
    """Обучает модель-кандидата и считает её метрики на test-месяце.

    Читает: splits/{train,valid,test}_proc.parquet из preproc_info.
    Пишет: черновик модели .staging_<version>.cbm в каталог моделей.
    latest.txt не трогает — решение о публикации принимает quality gate.

    Обучение в два прохода:
    1. train → early stopping по valid-месяцу, находим число деревьев;
    2. финальная модель с этим числом деревьев учится на train + valid,
       чтобы кандидат видел столько же месяцев, сколько прод-модель.
    Test-месяц не участвует ни в одном проходе: на нём сравниваются модели.

    Повторный запуск безопасен: каждый запуск пишет черновик с новой версией,
    существующие опубликованные модели и состояние окна не меняются.
    Возвращает версию, путь к черновику и метрики кандидата.
    """
    from catboost import CatBoostClassifier

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    x_train, y_train = split_xy(pd.read_parquet(preproc_info["train_proc_path"]))
    x_valid, y_valid = split_xy(pd.read_parquet(preproc_info["valid_proc_path"]))
    x_test, y_test = split_xy(pd.read_parquet(preproc_info["test_proc_path"]))

    probe = CatBoostClassifier(**CATBOOST_PARAMS)
    probe.fit(
        x_train,
        y_train,
        cat_features=CAT_FEATURES,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )
    best_iteration = int(probe.get_best_iteration() or CATBOOST_PARAMS["iterations"] - 1)
    logger.info("Early stopping по valid %s: лучшая итерация %s", preproc_info["valid_month"], best_iteration)

    final_params = {k: v for k, v in CATBOOST_PARAMS.items() if k != "early_stopping_rounds"}
    final_params["iterations"] = max(best_iteration + 1, 50)
    model = CatBoostClassifier(**final_params)
    model.fit(
        pd.concat([x_train, x_valid], ignore_index=True),
        pd.concat([y_train, y_valid], ignore_index=True),
        cat_features=CAT_FEATURES,
    )

    metrics = classification_metrics(y_test, model.predict_proba(x_test)[:, 1])
    logger.info("Кандидат на test %s: %s", preproc_info["test_month"], metrics)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging_model = MODELS_DIR / f".staging_{version}.cbm"
    model.save_model(str(staging_model))
    logger.info("Черновик модели сохранён: %s", staging_model)
    return {
        "version": version,
        "staging_model": str(staging_model),
        "dataset_path": preproc_info["dataset_path"],
        "batch_month": preproc_info["batch_month"],
        "train_months": preproc_info["train_months"],
        "valid_month": preproc_info["valid_month"],
        "test_month": preproc_info["test_month"],
        "test_proc_path": preproc_info["test_proc_path"],
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "best_iteration": best_iteration,
        "cleaning": preproc_info.get("cleaning"),
        "candidate_metrics": metrics,
    }
