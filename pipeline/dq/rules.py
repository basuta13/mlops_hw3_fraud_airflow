"""Правила Data Quality для витрины labeled_transactions.

Каждое правило — чистая функция над DataFrame, которая возвращает словарь
с результатом. Правила не ходят в БД, поэтому их легко тестировать.

Severity:
- critical — данные непригодны для обучения, DAG DQ падает на задаче dq_gate;
- warning  — подозрительно, но не блокирует; видно в отчёте.
"""

import numpy as np
import pandas as pd

from pipeline.settings import (
    DQ_ALLOWED_GENDERS,
    DQ_AMOUNT_MAX,
    DQ_FRAUD_RATE_MAX,
    DQ_FRAUD_RATE_MIN,
    DQ_MAX_AMOUNT_OUTLIER_SHARE,
    DQ_MAX_DUPLICATE_SHARE,
    DQ_MAX_OPTIONAL_NULL_SHARE,
    DQ_MIN_MONTH_ROWS,
    DQ_PSI_MAX,
)

KEY_COLUMNS = ["transaction_time", "amount", "target", "lat", "lon", "merchant_lat", "merchant_lon", "cat_id"]
NUMERIC_COLUMNS = ["amount", "lat", "lon", "merchant_lat", "merchant_lon", "population_city", "post_code", "target"]
EXAMPLES_LIMIT = 5


def _result(rule_id, title, severity, passed, observed, threshold, details="", examples=None):
    """Единый формат результата правила (сериализуется в XCom и JSON)."""
    return {
        "rule_id": rule_id,
        "title": title,
        "severity": severity,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "threshold": threshold,
        "details": details,
        "examples": examples or [],
    }


def _examples(frame: pd.DataFrame, mask: pd.Series) -> list[dict]:
    """До EXAMPLES_LIMIT строк, нарушивших правило, в JSON-совместимом виде."""
    sample = frame.loc[mask].head(EXAMPLES_LIMIT)
    return [
        {k: (None if pd.isna(v) else str(v)) for k, v in row.items()}
        for row in sample.to_dict(orient="records")
    ]


def _pct(value: float) -> str:
    return f"{value * 100:.3f}%"


# ---------------- Структура и полнота ----------------

def rule_schema(frame: pd.DataFrame, expected_columns: list[str]) -> dict:
    """DQ-01. Все колонки контракта на месте, числовые колонки имеют числовой тип."""
    missing = [c for c in expected_columns if c not in frame.columns]
    not_numeric = [
        c for c in NUMERIC_COLUMNS
        if c in frame.columns and not pd.api.types.is_numeric_dtype(frame[c])
    ]
    problems = []
    if missing:
        problems.append(f"нет колонок: {missing}")
    if not_numeric:
        problems.append(f"не числовые: {not_numeric}")
    return _result(
        "DQ-01", "Схема витрины соответствует контракту", "critical",
        not problems,
        observed=f"{len(frame.columns)} колонок" + (f"; {'; '.join(problems)}" if problems else ""),
        threshold=f"{len(expected_columns)} обязательных колонок, числовые: {', '.join(NUMERIC_COLUMNS)}",
    )


def rule_completeness(frame: pd.DataFrame) -> dict:
    """DQ-02. Нет пропусков в ключевых колонках, в остальных — не больше порога."""
    nulls = frame.isna().mean()
    key_bad = {c: float(nulls.get(c, 1.0)) for c in KEY_COLUMNS if nulls.get(c, 1.0) > 0}
    other = [c for c in frame.columns if c not in KEY_COLUMNS]
    other_bad = {c: float(nulls[c]) for c in other if nulls[c] > DQ_MAX_OPTIONAL_NULL_SHARE}
    mask = frame[[c for c in KEY_COLUMNS if c in frame.columns]].isna().any(axis=1)
    observed = "пропусков нет" if not (key_bad or other_bad) else ", ".join(
        f"{c}: {_pct(v)}" for c, v in {**key_bad, **other_bad}.items()
    )
    return _result(
        "DQ-02", "Полнота: нет пропусков в ключевых полях", "critical",
        not key_bad and not other_bad,
        observed=observed,
        threshold=f"0% в {', '.join(KEY_COLUMNS)}; ≤ {_pct(DQ_MAX_OPTIONAL_NULL_SHARE)} в остальных",
        examples=_examples(frame, mask),
    )


def rule_volume_and_continuity(frame: pd.DataFrame, monthly_counts: list[dict]) -> dict:
    """DQ-03. В свежем месяце достаточно строк и в витрине нет пропущенных месяцев."""
    months = pd.PeriodIndex([m["month"] for m in monthly_counts], freq="M")
    full_range = pd.period_range(months.min(), months.max(), freq="M")
    gaps = sorted(set(full_range.astype(str)) - set(months.astype(str)))
    enough = len(frame) >= DQ_MIN_MONTH_ROWS
    observed = f"{len(frame)} строк в свежем месяце; месяцев в витрине {len(months)}"
    if gaps:
        observed += f"; пропущены месяцы: {gaps}"
    return _result(
        "DQ-03", "Объём свежего месяца и непрерывность месяцев", "critical",
        enough and not gaps,
        observed=observed,
        threshold=f"≥ {DQ_MIN_MONTH_ROWS} строк; без пропусков между {months.min()} и {months.max()}",
    )


# ---------------- Валидность значений ----------------

def rule_target_domain(frame: pd.DataFrame) -> dict:
    """DQ-04. Метка target принимает только значения 0 и 1."""
    mask = ~frame["target"].isin([0, 1])
    values = sorted(map(str, frame["target"].dropna().unique()))[:10]
    return _result(
        "DQ-04", "Метка target ∈ {0, 1}", "critical",
        not mask.any(),
        observed=f"значения: {values}; некорректных строк: {int(mask.sum())}",
        threshold="только 0 и 1",
        examples=_examples(frame, mask),
    )


def rule_amount_range(frame: pd.DataFrame) -> dict:
    """DQ-05. Сумма строго положительна, доля аномально больших сумм мала."""
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    non_positive = amount <= 0
    outliers = amount > DQ_AMOUNT_MAX
    outlier_share = float(outliers.mean()) if len(frame) else 0.0
    return _result(
        "DQ-05", "Сумма транзакции в допустимом диапазоне", "critical",
        not non_positive.any() and outlier_share <= DQ_MAX_AMOUNT_OUTLIER_SHARE,
        observed=(
            f"min={amount.min():.2f}, max={amount.max():.2f}; ≤0: {int(non_positive.sum())}; "
            f"> {DQ_AMOUNT_MAX:g}: {_pct(outlier_share)}"
        ),
        threshold=f"amount > 0 у всех; доля amount > {DQ_AMOUNT_MAX:g} ≤ {_pct(DQ_MAX_AMOUNT_OUTLIER_SHARE)}",
        examples=_examples(frame, non_positive | outliers),
    )


def rule_geo_range(frame: pd.DataFrame) -> dict:
    """DQ-06. Координаты клиента и мерчанта валидны, население города > 0."""
    mask = (
        ~frame["lat"].between(-90, 90)
        | ~frame["merchant_lat"].between(-90, 90)
        | ~frame["lon"].between(-180, 180)
        | ~frame["merchant_lon"].between(-180, 180)
        | ~(frame["population_city"] > 0)
    )
    return _result(
        "DQ-06", "Координаты и население города валидны", "critical",
        not mask.any(),
        observed=f"некорректных строк: {int(mask.sum())}",
        threshold="lat ∈ [-90, 90], lon ∈ [-180, 180] (клиент и мерчант), population_city > 0",
        examples=_examples(frame, mask),
    )


def rule_categorical_domain(frame: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """DQ-07. gender из справочника, в cat_id нет категорий, которых не было в референсе."""
    bad_gender = ~frame["gender"].isin(DQ_ALLOWED_GENDERS)
    known = set(reference["cat_id"].dropna().unique()) if len(reference) else set(frame["cat_id"].dropna().unique())
    new_cats = sorted(set(frame["cat_id"].dropna().unique()) - known)
    mask = bad_gender | frame["cat_id"].isin(new_cats)
    return _result(
        "DQ-07", "Категориальные поля из известного справочника", "warning",
        not bad_gender.any() and not new_cats,
        observed=f"неизвестный gender: {int(bad_gender.sum())}; новые cat_id: {new_cats or 'нет'}",
        threshold=f"gender ∈ {list(DQ_ALLOWED_GENDERS)}; cat_id ⊆ категорий предыдущего месяца ({len(known)} шт.)",
        examples=_examples(frame, mask),
    )


# ---------------- Согласованность и распределения ----------------

def rule_duplicates(frame: pd.DataFrame) -> dict:
    """DQ-08. Доля полных дубликатов строк не превышает порог."""
    mask = frame.duplicated(keep="first")
    share = float(mask.mean()) if len(frame) else 0.0
    return _result(
        "DQ-08", "Доля полных дубликатов", "warning",
        share <= DQ_MAX_DUPLICATE_SHARE,
        observed=f"{int(mask.sum())} дублей ({_pct(share)})",
        threshold=f"≤ {_pct(DQ_MAX_DUPLICATE_SHARE)}",
        examples=_examples(frame, mask),
    )


def rule_fraud_rate(frame: pd.DataFrame) -> dict:
    """DQ-09. Доля фрода в правдоподобном коридоре (защита от сломанной разметки)."""
    rate = float(frame["target"].mean()) if len(frame) else 0.0
    return _result(
        "DQ-09", "Доля фрода в ожидаемом коридоре", "critical",
        DQ_FRAUD_RATE_MIN <= rate <= DQ_FRAUD_RATE_MAX,
        observed=f"{_pct(rate)} ({int(frame['target'].sum())} из {len(frame)})",
        threshold=f"[{_pct(DQ_FRAUD_RATE_MIN)}, {_pct(DQ_FRAUD_RATE_MAX)}]",
    )


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI числового признака: бины — квантили референса."""
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI категориального признака по долям категорий."""
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e = np.clip(e.reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    a = np.clip(a.reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def rule_distribution_drift(frame: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """DQ-10. Распределения amount и cat_id не уехали относительно предыдущего месяца."""
    if reference.empty:
        return _result(
            "DQ-10", "Дрейф распределений (PSI)", "warning", True,
            observed="нет референсного месяца, проверка пропущена",
            threshold=f"PSI < {DQ_PSI_MAX}",
        )
    psi_amount = population_stability_index(reference["amount"], frame["amount"])
    psi_cat = categorical_psi(reference["cat_id"], frame["cat_id"])
    psi_hour = population_stability_index(
        pd.to_datetime(reference["transaction_time"]).dt.hour,
        pd.to_datetime(frame["transaction_time"]).dt.hour,
        bins=24,
    )
    worst = max(psi_amount, psi_cat, psi_hour)
    return _result(
        "DQ-10", "Дрейф распределений (PSI)", "warning",
        worst < DQ_PSI_MAX,
        observed=f"PSI amount={psi_amount:.4f}, cat_id={psi_cat:.4f}, hour={psi_hour:.4f}",
        threshold=f"PSI < {DQ_PSI_MAX} для каждого признака",
    )
