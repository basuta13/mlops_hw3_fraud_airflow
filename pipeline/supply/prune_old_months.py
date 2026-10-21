"""Очистка витрины: хранить только последние HISTORY_YEARS лет."""

import logging
from datetime import date

from dateutil.relativedelta import relativedelta

from pipeline.db import get_conn
from pipeline.settings import HISTORY_YEARS

logger = logging.getLogger(__name__)


def prune_old_months(batch_info: dict) -> dict:
    """Удаляет наблюдения старше HISTORY_YEARS лет относительно самого нового месяца.

    Читает: max(transaction_time) из labeled_transactions после дозаписи батча.
    Пишет: удаляет из labeled_transactions строки с transaction_time строго
    раньше границы «начало самого нового месяца минус HISTORY_YEARS лет».

    Повторный запуск безопасен: граница считается от данных, а не от счётчика
    запусков, поэтому второй DELETE с той же границей удаляет 0 строк.
    Возвращает месяц батча, границу хранения и число удалённых строк.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date_trunc('month', max(transaction_time))::date
                FROM labeled_transactions
                """
            )
            newest_month = cur.fetchone()[0]
            if newest_month is None:
                raise RuntimeError("Нечего чистить: витрина labeled_transactions пуста.")
            cutoff = newest_month - relativedelta(years=HISTORY_YEARS)
            cur.execute(
                "DELETE FROM labeled_transactions WHERE transaction_time < %s",
                (cutoff,),
            )
            deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "Глубина хранения %s лет от %s. Удалено строк старше %s: %s",
        HISTORY_YEARS,
        newest_month,
        cutoff,
        deleted,
    )
    return {
        "batch_month": batch_info["batch_month"],
        "newest_month": newest_month.isoformat() if isinstance(newest_month, date) else str(newest_month),
        "cutoff": cutoff.isoformat(),
        "deleted_rows": int(deleted),
    }
