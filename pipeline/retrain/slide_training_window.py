"""Сдвиг скользящего окна обучения на один месяц вперёд."""

import logging
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.settings import CURRENT_DATASET, PENDING_WINDOW, WINDOW_MONTHS

logger = logging.getLogger(__name__)


def build_window(current: pd.DataFrame, batch: pd.DataFrame, batch_month: date) -> pd.DataFrame:
    """Собирает окно из WINDOW_MONTHS месяцев, которое заканчивается batch_month.

    Из current выбрасываются строки batch_month (если они там уже есть после
    частично упавшего прогона), добавляется batch, затем остаются только месяцы
    [batch_month - (WINDOW_MONTHS - 1), batch_month]. Результат зависит только
    от batch_month, поэтому повторный вызов не сдвигает окно второй раз.
    """
    batch_period = pd.Period(batch_month, freq="M")
    first_period = pd.Period(batch_month - relativedelta(months=WINDOW_MONTHS - 1), freq="M")

    current = current.copy()
    batch = batch.copy()
    current["transaction_time"] = pd.to_datetime(current["transaction_time"])
    batch["transaction_time"] = pd.to_datetime(batch["transaction_time"])

    current = current.loc[current["transaction_time"].dt.to_period("M") != batch_period]
    combined = pd.concat([current, batch], ignore_index=True)
    periods = combined["transaction_time"].dt.to_period("M")
    window = combined.loc[(periods >= first_period) & (periods <= batch_period)]
    return window.sort_values("transaction_time").reset_index(drop=True)


def slide_training_window(batch_info: dict) -> dict:
    """Добавляет новый месяц к текущему окну и убирает самый старый.

    Читает: current.parquet (текущее окно) и parquet батча из batch_info.
    Пишет: pending_window.parquet. current.parquet НЕ трогает: его заменяет
    advance_training_window только после сохранения модели.

    Повторный запуск безопасен: окно строится как «последние WINDOW_MONTHS
    месяцев до batch_month включительно», а не «отрезать самый старый месяц».
    Даже если current.parquet уже содержит batch_month, результат тот же.
    Возвращает путь к новому окну и месяц добавленного батча.
    """
    if not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет исходного датасета {CURRENT_DATASET}")

    batch_month = date.fromisoformat(batch_info["batch_month"])
    current = pd.read_parquet(CURRENT_DATASET)
    batch = pd.read_parquet(batch_info["batch_path"])
    window = build_window(current, batch, batch_month)

    months = sorted(window["transaction_time"].dt.to_period("M").unique().astype(str))
    if len(months) != WINDOW_MONTHS:
        logger.warning("В окне %s месяцев вместо %s: %s", len(months), WINDOW_MONTHS, months)
    window.to_parquet(PENDING_WINDOW, index=False)
    logger.info(
        "Окно после сдвига: %s строк, месяцы %s.",
        len(window),
        months,
    )
    return {
        "dataset_path": str(PENDING_WINDOW),
        "batch_month": batch_month.isoformat(),
        "window_months": ",".join(months),
    }
