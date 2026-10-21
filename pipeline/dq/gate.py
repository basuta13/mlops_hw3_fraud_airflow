"""Финальная задача DQ: красный DAG, если нарушены critical-правила."""

import logging

from airflow.exceptions import AirflowFailException

logger = logging.getLogger(__name__)


def dq_gate(summary: dict) -> dict:
    """Падает без ретраев, если упало хотя бы одно critical-правило.

    Читает: сводку build_dq_report из XCom. Пишет: ничего.
    Запускается после публикации отчёта, поэтому отчёт есть даже при падении.
    Повторный запуск безопасен: результат определяется только сводкой.
    """
    if summary["failed_critical"]:
        raise AirflowFailException(
            f"DQ {summary['month']}: нарушены critical-правила {summary['failed_critical']}. "
            f"Отчёт: {summary['report_path']}"
        )
    if summary["failed_warning"]:
        logger.warning("DQ %s: предупреждения %s", summary["month"], summary["failed_warning"])
    logger.info("DQ %s: %s, отчёт %s", summary["month"], summary["status"], summary["report_path"])
    return summary
