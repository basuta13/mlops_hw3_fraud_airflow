"""DAG проверок качества данных витрины labeled_transactions.

Файл только связывает задачи. Логика каждой задачи лежит в pipeline.dq.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.dq.checks import check_consistency, check_structure, check_validity
from pipeline.dq.gate import dq_gate
from pipeline.dq.report import build_dq_report
from pipeline.dq.snapshot import extract_dq_snapshot


@dag(
    dag_id="data_quality_checks",
    description="Раз в 10 минут проверяет свежий месяц витрины по 10 DQ-правилам и публикует HTML-отчёт",
    schedule="*/10 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=10),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "data-quality"],
    doc_md="""
### Data Quality витрины

1. `extract_dq_snapshot` — свежий месяц, предыдущий месяц (референс) и объёмы по месяцам.
2. Три параллельные группы правил: `check_structure`, `check_validity`, `check_consistency`.
3. `build_dq_report` — HTML и JSON в `/opt/airflow/dq_reports` (рядом с `/opt/airflow/models`).
4. `dq_gate` — DAG красный, если нарушено хотя бы одно critical-правило.

Список правил и порогов — в README.
    """,
)
def data_quality_checks():
    """Связывает снимок витрины, группы проверок, отчёт и гейт."""
    snapshot = task(extract_dq_snapshot, task_id="extract_dq_snapshot")()
    structure = task(check_structure, task_id="check_structure")(snapshot)
    validity = task(check_validity, task_id="check_validity")(snapshot)
    consistency = task(check_consistency, task_id="check_consistency")(snapshot)
    summary = task(build_dq_report, task_id="build_dq_report")(snapshot, structure, validity, consistency)
    task(dq_gate, task_id="dq_gate", retries=0)(summary)


data_quality_checks()
