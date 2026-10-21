"""Препроцессинг транзакций: общий для онлайн-скоринга и для DAG переобучения.

Функция run_preproc вызывается и в fraud_detector (на одной транзакции из Kafka),
и в задаче preprocess_dataset DAG retrain_catboost (на всём окне обучения).
Поэтому все признаки stateless: считаются только по самой строке, без статистик
по выборке. Так в train и в проде признаки гарантированно считаются одинаково.

Набор признаков — надмножество признаков семинара (haversine, year, month, day,
hour, minute, weekend), чтобы исходная модель initial из образа тоже могла скорить.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
# Категории онлайн-покупок в датасете имеют суффикс _net (shopping_net, misc_net, ...).
ONLINE_SUFFIX = "_net"
# Ночное окно, в котором в датасете концентрируется фрод.
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}
DROP_COLUMNS = ["transaction_time", "name_1", "name_2", "street"]


def haversine(lat1, lon1, lat2, lon2):
    """Расстояние по дуге большого круга между клиентом и мерчантом, км."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """Направление от клиента на мерчанта в градусах [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _to_numeric(df: pd.DataFrame, columns) -> None:
    """Приводит колонки к числам на месте: из Kafka они могут прийти строками."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Календарные признаки и циклическое кодирование часа."""
    ts = pd.to_datetime(df["transaction_time"])
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["weekend"] = (ts.dt.dayofweek > 4).astype(int)
    df["is_night"] = ts.dt.hour.isin(NIGHT_HOURS).astype(int)
    # Час — циклический признак: 23:00 и 00:00 должны быть рядом.
    minutes_of_day = ts.dt.hour * 60 + ts.dt.minute
    df["time_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
    df["time_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки суммы: лог-масштаб, «круглость», ночная крупная покупка."""
    amount = df["amount"].astype(float)
    df["amount_log"] = np.log1p(amount.clip(lower=0))
    df["amount_cents"] = np.round((amount * 100) % 100).fillna(-1).astype(int)
    df["amount_is_round"] = (df["amount_cents"] == 0).astype(int)
    df["night_amount_log"] = df["amount_log"] * df["is_night"]
    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Расстояние и направление клиент → мерчант, лог населения города."""
    df["haversine"] = haversine(df["lat"], df["lon"], df["merchant_lat"], df["merchant_lon"])
    df["haversine_log"] = np.log1p(df["haversine"])
    df["merchant_bearing"] = bearing(df["lat"], df["lon"], df["merchant_lat"], df["merchant_lon"])
    df["city_pop_log"] = np.log1p(df["population_city"].astype(float).clip(lower=0))
    return df


def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Онлайн/офлайн категория покупки."""
    df["is_online"] = df["cat_id"].astype(str).str.endswith(ONLINE_SUFFIX).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Строит все признаки модели и удаляет сырые колонки, не нужные модели."""
    df_ = df.copy()
    _to_numeric(df_, ["amount", "lat", "lon", "merchant_lat", "merchant_lon", "population_city"])
    df_ = add_time_features(df_)
    df_ = add_amount_features(df_)
    df_ = add_geo_features(df_)
    df_ = add_category_features(df_)
    return df_.drop(columns=[c for c in DROP_COLUMNS if c in df_.columns])


def run_preproc(input_df: pd.DataFrame) -> pd.DataFrame:
    """Главная функция препроцессинга: сырые транзакции → матрица признаков."""
    output_df = add_features(input_df)
    logger.info("Added new features. Output shape: %s", output_df.shape)
    return output_df
