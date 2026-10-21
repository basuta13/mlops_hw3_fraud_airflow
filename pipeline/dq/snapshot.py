"""Снимок витрины для проверок качества данных."""

import json
import logging
from datetime import datetime, timezone

import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive
from pipeline.settings import DQ_WORK_DIR

logger = logging.getLogger(__name__)


def _read_month(conn, month_start) -> pd.DataFrame:
    """Все строки календарного месяца month_start со всеми колонками витрины."""
    return pd.read_sql(
        "SELECT * FROM labeled_transactions WHERE transaction_time >= %s AND transaction_time < %s",
        conn,
        params=(month_start, month_end_exclusive(month_start)),
    )


def extract_dq_snapshot() -> dict:
    """Снимает с витрины данные, которые проверяют задачи DQ.

    Читает из labeled_transactions:
    - самый свежий месяц (его только что дописал supply_labeled_batch) — проверяемый;
    - предыдущий месяц — референс для правил дрейфа и справочника категорий;
    - число строк по каждому месяцу — для правила непрерывности.
    Пишет: dq/current.parquet, dq/reference.parquet, dq/monthly_counts.json.

    Повторный запуск безопасен: задача только читает витрину и перезаписывает
    свои рабочие файлы, на витрину и модели не влияет.
    """
    DQ_WORK_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        counts = pd.read_sql(
            """
            SELECT date_trunc('month', transaction_time)::date AS month, count(*) AS rows
            FROM labeled_transactions
            GROUP BY 1
            ORDER BY 1
            """,
            conn,
        )
        if counts.empty:
            raise RuntimeError("Витрина labeled_transactions пуста. Сначала должен отработать data_loader.")
        current_month = pd.Timestamp(counts["month"].max()).date()
        reference_month = current_month - relativedelta(months=1)
        current = _read_month(conn, current_month)
        reference = _read_month(conn, reference_month)
    finally:
        conn.close()

    current_path = DQ_WORK_DIR / "current.parquet"
    reference_path = DQ_WORK_DIR / "reference.parquet"
    counts_path = DQ_WORK_DIR / "monthly_counts.json"
    current.to_parquet(current_path, index=False)
    reference.to_parquet(reference_path, index=False)
    counts_path.write_text(
        json.dumps(
            [{"month": pd.Timestamp(m).date().isoformat(), "rows": int(r)} for m, r in zip(counts["month"], counts["rows"])],
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "DQ снимок: месяц %s (%s строк), референс %s (%s строк), всего месяцев %s",
        current_month, len(current), reference_month, len(reference), len(counts),
    )
    return {
        "current_path": str(current_path),
        "reference_path": str(reference_path),
        "monthly_counts_path": str(counts_path),
        "current_month": current_month.isoformat(),
        "reference_month": reference_month.isoformat(),
        "current_rows": int(len(current)),
        "reference_rows": int(len(reference)),
        "expected_columns": LABELED_COLUMNS,
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
    }
