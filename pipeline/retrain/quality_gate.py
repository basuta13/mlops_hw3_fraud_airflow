"""Quality gate: новая модель выкатывается, только если она не хуже прод-модели.

Обе модели оцениваются на одном и том же test-месяце — самом свежем месяце окна.
Ни одна из них этот месяц не видела: кандидат учился на более ранних месяцах,
прод-модель — на окне, сдвинутом ещё на месяц назад. Поэтому сравнение честное
(out-of-time) и отражает, какая модель лучше скорит «следующий» месяц.
"""

import logging
import os

import pandas as pd

from pipeline.retrain.metrics import classification_metrics, features_for_model, split_xy
from pipeline.settings import BOOTSTRAP_VERSION, MIN_PR_AUC_GAIN, MODELS_DIR, ROC_AUC_TOLERANCE

logger = logging.getLogger(__name__)

PUBLISH_TASK_ID = "publish_model"
REJECT_TASK_ID = "reject_model"


def read_prod_version() -> str | None:
    """Возвращает версию из latest.txt или None, если прод-модели ещё нет."""
    latest = MODELS_DIR / "latest.txt"
    if not latest.exists():
        return None
    version = latest.read_text(encoding="utf-8").strip()
    return version or None


def decide_promotion(candidate: dict, prod: dict | None, prod_version: str | None) -> tuple[bool, str]:
    """Правило выкатки. Возвращает (публиковать ли, человекочитаемая причина).

    1. Нет прод-модели или она не читается → публикуем кандидата.
    2. Прод — бандл-модель initial из образа → публикуем: она обучена вне пайплайна
       на неизвестном окне (возможно, включая test-месяцы), её метрики несопоставимы.
    3. У кандидата AUC не определён (в test одна метка) → не публикуем, оставляем прод.
    4. Иначе публикуем, если PR-AUC кандидата >= PR-AUC прода + MIN_PR_AUC_GAIN
       и ROC-AUC кандидата >= ROC-AUC прода - ROC_AUC_TOLERANCE.
    """
    if prod_version is None or prod is None:
        return True, "прод-модели нет или её не удалось оценить"
    if prod_version == BOOTSTRAP_VERSION:
        return True, f"в проде бандл-модель '{BOOTSTRAP_VERSION}', первая модель пайплайна выкатывается без сравнения"
    if candidate.get("pr_auc") is None or candidate.get("roc_auc") is None:
        return False, "у кандидата не определены PR-AUC/ROC-AUC (в test одна метка)"
    if prod.get("pr_auc") is None or prod.get("roc_auc") is None:
        return True, "у прод-модели не определены метрики на test"

    pr_ok = candidate["pr_auc"] >= prod["pr_auc"] + MIN_PR_AUC_GAIN
    roc_ok = candidate["roc_auc"] >= prod["roc_auc"] - ROC_AUC_TOLERANCE
    detail = (
        f"PR-AUC {candidate['pr_auc']:.4f} vs {prod['pr_auc']:.4f} (нужно >= +{MIN_PR_AUC_GAIN}), "
        f"ROC-AUC {candidate['roc_auc']:.4f} vs {prod['roc_auc']:.4f} (допуск -{ROC_AUC_TOLERANCE})"
    )
    if pr_ok and roc_ok:
        return True, "кандидат не хуже прода: " + detail
    return False, "кандидат хуже прода: " + detail


def evaluate_prod_model(train_info: dict) -> dict:
    """Оценивает текущую прод-модель на test-месяце кандидата и принимает решение.

    Читает: latest.txt и my_catboost_<prod>.cbm из каталога моделей,
    splits/test_proc.parquet, метрики кандидата из train_info.
    Пишет: ничего, только возвращает XCom с метриками и решением.

    Повторный запуск безопасен: задача только читает файлы. Если между запусками
    прод-модель сменилась, решение честно пересчитается относительно новой.
    Возвращает train_info, дополненный prod_version, prod_metrics, promote, reason.
    """
    from catboost import CatBoostClassifier

    prod_version = read_prod_version()
    prod_metrics = None
    if prod_version is not None:
        path = MODELS_DIR / f"my_catboost_{prod_version}.cbm"
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            model = CatBoostClassifier()
            model.load_model(str(path))
            test = pd.read_parquet(train_info["test_proc_path"])
            x_test, y_test = split_xy(test)
            proba = model.predict_proba(features_for_model(model, x_test))[:, 1]
            prod_metrics = classification_metrics(y_test, proba)
            logger.info("Прод-модель %s на test %s: %s", prod_version, train_info["test_month"], prod_metrics)
        except Exception:  # noqa: BLE001 — битая прод-модель не должна блокировать выкатку
            logger.exception("Не удалось оценить прод-модель %s (%s)", prod_version, path)

    promote, reason = decide_promotion(train_info["candidate_metrics"], prod_metrics, prod_version)
    logger.info("Решение quality gate: %s — %s", "PUBLISH" if promote else "REJECT", reason)
    return {
        **train_info,
        "prod_version": prod_version,
        "prod_metrics": prod_metrics,
        "promote": promote,
        "reason": reason,
    }


def choose_deployment(evaluation: dict) -> str:
    """Ветвление DAG: возвращает task_id публикации или отклонения кандидата.

    Читает только XCom задачи evaluate_prod_model. Ничего не пишет,
    поэтому повтор всегда даёт то же решение для тех же метрик.
    """
    return PUBLISH_TASK_ID if evaluation["promote"] else REJECT_TASK_ID
