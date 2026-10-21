"""Задачи DAG data_quality_checks: группы правил над снимком витрины.

Каждая задача читает файлы снимка, применяет свою группу правил из
pipeline.dq.rules и возвращает список результатов в XCom. Задачи ничего
не пишут ни в витрину, ни на диск, поэтому их повторный запуск безопасен.
"""

import json
import logging

import pandas as pd

from pipeline.dq import rules

logger = logging.getLogger(__name__)


def _load(snapshot: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = pd.read_parquet(snapshot["current_path"])
    reference = pd.read_parquet(snapshot["reference_path"])
    return current, reference


def _log(group: str, results: list[dict]) -> list[dict]:
    for r in results:
        logger.info("[%s] %s %s: %s", group, r["rule_id"], r["status"].upper(), r["observed"])
    return results


def check_structure(snapshot: dict) -> list[dict]:
    """Структура и полнота: DQ-01 схема, DQ-02 пропуски, DQ-03 объём и непрерывность.

    Читает: dq/current.parquet и dq/monthly_counts.json из снимка.
    Пишет: ничего, возвращает результаты правил. Повтор безопасен.
    """
    current, _ = _load(snapshot)
    with open(snapshot["monthly_counts_path"], encoding="utf-8") as fh:
        monthly_counts = json.load(fh)
    return _log("structure", [
        rules.rule_schema(current, snapshot["expected_columns"]),
        rules.rule_completeness(current),
        rules.rule_volume_and_continuity(current, monthly_counts),
    ])


def check_validity(snapshot: dict) -> list[dict]:
    """Валидность значений: DQ-04 target, DQ-05 сумма, DQ-06 гео, DQ-07 справочники.

    Читает: dq/current.parquet и dq/reference.parquet.
    Пишет: ничего, возвращает результаты правил. Повтор безопасен.
    """
    current, reference = _load(snapshot)
    return _log("validity", [
        rules.rule_target_domain(current),
        rules.rule_amount_range(current),
        rules.rule_geo_range(current),
        rules.rule_categorical_domain(current, reference),
    ])


def check_consistency(snapshot: dict) -> list[dict]:
    """Согласованность и распределения: DQ-08 дубли, DQ-09 доля фрода, DQ-10 дрейф.

    Читает: dq/current.parquet и dq/reference.parquet.
    Пишет: ничего, возвращает результаты правил. Повтор безопасен.
    """
    current, reference = _load(snapshot)
    return _log("consistency", [
        rules.rule_duplicates(current),
        rules.rule_fraud_rate(current),
        rules.rule_distribution_drift(current, reference),
    ])
