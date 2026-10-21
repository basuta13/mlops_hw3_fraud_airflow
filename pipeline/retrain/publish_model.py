"""Публикация или отклонение модели-кандидата."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from pipeline.settings import MODELS_DIR, REJECTED_DIR

logger = logging.getLogger(__name__)


def _metrics_payload(evaluation: dict, status: str) -> dict:
    """Карточка модели: окно, метрики кандидата и прода, решение gate."""
    keys = (
        "version", "batch_month", "train_months", "valid_month", "test_month",
        "train_rows", "valid_rows", "best_iteration", "cleaning",
        "candidate_metrics", "prod_version", "prod_metrics", "reason",
    )
    payload = {key: evaluation.get(key) for key in keys}
    payload["status"] = status
    payload["decided_at"] = datetime.now(timezone.utc).isoformat()
    return payload


def _move_staging(evaluation: dict, target: Path) -> None:
    """Переносит черновик .cbm в target. Если уже перенесён — ничего не делает."""
    staging = Path(evaluation["staging_model"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        os.replace(staging, target)
    elif not target.exists():
        raise FileNotFoundError(f"Нет ни черновика {staging}, ни файла {target}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def publish_model(evaluation: dict) -> str:
    """Выкатывает кандидата: кладёт бинарник в каталог моделей и переключает latest.txt.

    Читает: черновик .staging_<version>.cbm и XCom evaluate_prod_model.
    Пишет: my_catboost_<version>.cbm, my_catboost_<version>.metrics.json
    и атомарно (через os.replace) latest.txt. fraud_detector перечитывает
    latest.txt на каждом сообщении и подхватывает модель без перезапуска.

    Повторный запуск безопасен: если черновик уже переименован, повторно
    переписываются только metrics.json и latest.txt с той же версией.
    Указатель окна здесь НЕ двигается — это делает advance_training_window.
    Возвращает опубликованную версию.
    """
    version = evaluation["version"]
    final_model = MODELS_DIR / f"my_catboost_{version}.cbm"
    _move_staging(evaluation, final_model)
    _write_json_atomic(MODELS_DIR / f"my_catboost_{version}.metrics.json", _metrics_payload(evaluation, "published"))

    latest_tmp = MODELS_DIR / f".latest.{version}.txt"
    latest_tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(latest_tmp, MODELS_DIR / "latest.txt")
    logger.info("Опубликована модель %s (%s). Причина: %s", version, final_model, evaluation["reason"])
    return version


def reject_model(evaluation: dict) -> str:
    """Не выкатывает кандидата: сохраняет его в rejected/ для разбора, прод не меняется.

    Читает: черновик .staging_<version>.cbm и XCom evaluate_prod_model.
    Пишет: rejected/my_catboost_<version>.cbm и rejected/my_catboost_<version>.metrics.json.
    latest.txt не трогает, сервис скоринга продолжает работать на прод-модели.

    Повторный запуск безопасен: перенос черновика идемпотентен, json перезаписывается.
    Месяц всё равно считается обработанным (окно сдвинется в advance_training_window),
    иначе DAG бесконечно переобучался бы на одном и том же месяце.
    Возвращает версию отклонённого кандидата.
    """
    version = evaluation["version"]
    _move_staging(evaluation, REJECTED_DIR / f"my_catboost_{version}.cbm")
    _write_json_atomic(REJECTED_DIR / f"my_catboost_{version}.metrics.json", _metrics_payload(evaluation, "rejected"))
    logger.warning(
        "Кандидат %s отклонён, в проде остаётся %s. Причина: %s",
        version, evaluation.get("prod_version"), evaluation["reason"],
    )
    return version
