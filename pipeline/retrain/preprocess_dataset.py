"""Препроцессинг train / valid / test для обучения.

Признаки строит run_preproc из fraud_detector/src — та же функция, что и
в онлайн-скоринге. Поверх неё здесь есть очистка, которая нужна только
при обучении (в проде строку нельзя «выбросить», её нужно отскорить).
"""

import logging
import sys

import pandas as pd

from pipeline.settings import FRAUD_SRC, LABEL, SPLITS_DIR

logger = logging.getLogger(__name__)

REQUIRED_NOT_NULL = ["transaction_time", "amount", "lat", "lon", "merchant_lat", "merchant_lon"]


def clean_training_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Убирает строки, на которых нельзя честно учиться.

    - нет метки или метка не 0/1;
    - пустые время, сумма или координаты;
    - сумма <= 0;
    - координаты вне допустимых диапазонов;
    - полные дубликаты строк (в витрине они появляются при пересэмплировании).
    Возвращает очищенный frame и счётчики удалённых строк по причинам.
    """
    stats = {"input": int(len(frame))}
    df = frame.copy()

    mask = df[LABEL].isin([0, 1])
    stats["bad_label"] = int((~mask).sum())
    df = df.loc[mask]

    mask = df[REQUIRED_NOT_NULL].notna().all(axis=1)
    stats["nulls"] = int((~mask).sum())
    df = df.loc[mask]

    mask = df["amount"] > 0
    stats["non_positive_amount"] = int((~mask).sum())
    df = df.loc[mask]

    mask = (
        df["lat"].between(-90, 90)
        & df["merchant_lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
        & df["merchant_lon"].between(-180, 180)
    )
    stats["bad_coords"] = int((~mask).sum())
    df = df.loc[mask]

    before = len(df)
    df = df.drop_duplicates()
    stats["duplicates"] = int(before - len(df))
    stats["output"] = int(len(df))
    return df.reset_index(drop=True), stats


def preprocess_dataset(split_info: dict) -> dict:
    """Чистит train/valid, применяет run_preproc ко всем трём частям.

    Читает: splits/{train,valid,test}_raw.parquet из split_info.
    Пишет: splits/{train,valid,test}_proc.parquet.
    В test убираются только строки без корректной метки: на нём сравниваются
    модели, и он должен быть похож на реальный поток.

    Повторный запуск безопасен: функция чистая, файлы перезаписываются,
    БД и каталог моделей не меняются.
    Возвращает пути к обработанным частям, статистику очистки и метаданные месяцев.
    """
    if FRAUD_SRC not in sys.path:
        sys.path.insert(0, FRAUD_SRC)
    from preprocessing import run_preproc

    result = {key: split_info[key] for key in (
        "dataset_path", "batch_month", "train_months", "valid_month", "test_month",
    )}
    cleaning = {}
    for name in ("train", "valid", "test"):
        raw = pd.read_parquet(split_info[f"{name}_raw_path"])
        if name == "test":
            mask = raw[LABEL].isin([0, 1])
            cleaned, stats = raw.loc[mask], {"input": len(raw), "bad_label": int((~mask).sum())}
        else:
            cleaned, stats = clean_training_rows(raw)
        cleaning[name] = stats
        processed = run_preproc(cleaned)
        path = SPLITS_DIR / f"{name}_proc.parquet"
        processed.to_parquet(path, index=False)
        result[f"{name}_proc_path"] = str(path)
        logger.info("%s: очистка %s, после препроцессинга %s", name, stats, processed.shape)
    result["cleaning"] = cleaning
    return result
