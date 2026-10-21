"""Метрики качества и подготовка признаков под конкретную модель CatBoost."""

import pandas as pd

from pipeline.settings import CAT_FEATURES, LABEL, THRESHOLD


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет target от признаков и приводит категориальные колонки к строкам.

    Категории совпадают с теми, что ожидает онлайн-скоринг после run_preproc.
    """
    if LABEL not in frame.columns:
        raise ValueError(f"В датасете нет колонки {LABEL}")
    features = frame.drop(columns=[LABEL]).copy()
    for column in CAT_FEATURES:
        if column not in features.columns:
            raise ValueError(f"После препроцессинга нет категориального признака {column}")
        features[column] = features[column].astype(str)
    return features, frame[LABEL].astype(int)


def features_for_model(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Выбирает из frame признаки в порядке model.feature_names_.

    Так же поступает fraud_detector/src/scorer.py, поэтому прод-модель
    на test-месяце оценивается ровно на тех признаках, на которых она скорит.
    """
    names = list(model.feature_names_)
    missing = [column for column in names if column not in frame.columns]
    if missing:
        raise ValueError(f"Во входных данных нет признаков модели: {missing}")
    selected = frame[names].copy()
    for idx in model.get_cat_feature_indices():
        column = names[idx]
        if not pd.api.types.is_numeric_dtype(selected[column]):
            selected[column] = selected[column].astype(str)
    return selected


def classification_metrics(y_true: pd.Series, proba) -> dict:
    """Считает PR-AUC, ROC-AUC, F1, precision и recall при пороге THRESHOLD.

    Если в выборке одна метка, AUC-метрики не определены и возвращаются как None.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (proba > THRESHOLD).astype(int)
    both_classes = pd.Series(y_true).nunique() > 1
    return {
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "rows": int(len(y_true)),
        "fraud_rate": float(pd.Series(y_true).mean()) if len(y_true) else None,
    }
