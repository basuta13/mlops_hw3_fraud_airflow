"""Дозапись подготовленного месячного батча в размеченную витрину."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.db import get_conn, replace_month_batch

logger = logging.getLogger(__name__)


def append_labeled_batch(batch_info: dict) -> dict:
    """Дописывает parquet батча нового месяца в labeled_transactions.

    Читает: parquet по пути batch_info["batch_path"] (результат sample_next_month).
    Пишет: строки месяца batch_info["batch_month"] в таблицу labeled_transactions.

    Повторный запуск безопасен: в одной транзакции сначала удаляются строки
    этого месяца, затем вставляется батч. Если задача упала после COMMIT
    и Airflow её перезапустил, месяц не задублируется. Месяц батча всегда
    новее всех исторических данных, поэтому DELETE не затрагивает старые строки.
    Возвращает месяц батча и число записанных строк.
    """
    batch_path = Path(batch_info["batch_path"])
    if not batch_path.exists():
        raise FileNotFoundError(f"Нет файла батча для загрузки: {batch_path}")

    frame = pd.read_parquet(batch_path)
    month_start = date.fromisoformat(batch_info["batch_month"])
    conn = get_conn()
    try:
        replaced = replace_month_batch(conn, frame, month_start)
    finally:
        conn.close()
    if replaced:
        logger.warning("Месяц %s уже был в витрине (%s строк) и перезаписан.", month_start, replaced)
    logger.info("Месяц %s добавлен в витрину, строк: %s", month_start, len(frame))
    return {
        "batch_path": batch_info["batch_path"],
        "batch_month": batch_info["batch_month"],
        "rows": int(len(frame)),
    }
