"""Фиксация нового окна обучения после сохранения модели."""

import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.settings import CURRENT_DATASET

logger = logging.getLogger(__name__)


def advance_training_window(evaluation: dict) -> dict:
    """Делает новое окно текущим и сдвигает указатель месяца в training_state.

    Запускается последней, после publish_model ИЛИ reject_model (trigger rule
    none_failed_min_one_success): к этому моменту модель-кандидат уже сохранена
    на диск — либо опубликована, либо отложена в rejected/.

    Читает: pending_window.parquet из evaluation["dataset_path"].
    Пишет: заменяет current.parquet и обновляет строку training_state.

    Повторный запуск безопасен: если pending уже перенесён, файл не трогается,
    а training_state получает абсолютные значения, вычисленные из batch_month
    (window_end = batch_month), поэтому второй вызов окно не сдвигает.
    Если упала публикация, эта задача не запускается и месяц не теряется.
    """
    pending_window = Path(evaluation["dataset_path"])
    if pending_window.exists():
        os.replace(pending_window, CURRENT_DATASET)
        logger.info("Окно обучения обновлено: %s", CURRENT_DATASET)
    elif not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет нового окна обучения: {pending_window}")
    else:
        logger.info("Окно уже перенесено в %s, повторно не сдвигаем.", CURRENT_DATASET)

    batch_month = date.fromisoformat(evaluation["batch_month"])
    conn = get_conn()
    try:
        advance_training_state(conn, batch_month)
    finally:
        conn.close()
    return {
        "batch_month": evaluation["batch_month"],
        "promoted": bool(evaluation["promote"]),
        "version": evaluation["version"],
    }
