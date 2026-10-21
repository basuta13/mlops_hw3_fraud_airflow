"""Временной split окна: train / validation / test по месяцам."""

import logging

import pandas as pd

from pipeline.settings import SPLITS_DIR

logger = logging.getLogger(__name__)


def split_by_month(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит окно по времени, без перемешивания.

    test — самый свежий месяц, valid — предыдущий месяц (для early stopping),
    train — все более ранние месяцы. Случайного split здесь нет.
    """
    frame = frame.copy()
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"])
    frame = frame.sort_values("transaction_time")
    periods = frame["transaction_time"].dt.to_period("M")
    months = sorted(periods.unique())
    if len(months) < 3:
        raise RuntimeError(f"Для train/valid/test нужно минимум 3 месяца, в окне: {months}")
    test_month, valid_month = months[-1], months[-2]
    train = frame.loc[periods < valid_month]
    valid = frame.loc[periods == valid_month]
    test = frame.loc[periods == test_month]
    return train, valid, test


def time_train_test_split(window_info: dict) -> dict:
    """Делит окно по времени: ранние месяцы — train, предпоследний — valid, свежий — test.

    Читает: parquet окна из window_info["dataset_path"] (pending_window.parquet).
    Пишет: splits/train_raw.parquet, splits/valid_raw.parquet, splits/test_raw.parquet.
    Колонка времени остаётся: препроцессинг сам строит календарные признаки.

    Повторный запуск безопасен: результат детерминирован входным окном,
    файлы просто перезаписываются, состояние в БД не меняется.
    Test-месяц не участвует ни в обучении, ни в early stopping: на нём
    сравниваются новая и прод-модель в quality gate.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(window_info["dataset_path"])
    train, valid, test = split_by_month(frame)
    if train.empty or valid.empty or test.empty:
        raise RuntimeError(
            f"Пустая часть split: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    paths = {}
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        path = SPLITS_DIR / f"{name}_raw.parquet"
        part.to_parquet(path, index=False)
        paths[f"{name}_raw_path"] = str(path)

    def months_of(part):
        return ",".join(sorted(part["transaction_time"].dt.to_period("M").unique().astype(str)))

    info = {
        **paths,
        "dataset_path": window_info["dataset_path"],
        "batch_month": window_info["batch_month"],
        "train_months": months_of(train),
        "valid_month": months_of(valid),
        "test_month": months_of(test),
    }
    logger.info(
        "Train %s (%s строк), valid %s (%s), test %s (%s).",
        info["train_months"], len(train), info["valid_month"], len(valid),
        info["test_month"], len(test),
    )
    return info
