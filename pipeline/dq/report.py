"""HTML-отчёт по проверкам качества данных."""

import html
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from pipeline.settings import DQ_REPORTS_DIR

logger = logging.getLogger(__name__)

STYLE = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#1f2328;background:#fff}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px}
.meta{color:#59636e;font-size:13px;margin-bottom:16px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.card{border:1px solid #d1d9e0;border-radius:8px;padding:10px 16px;min-width:120px}
.card b{display:block;font-size:22px}
.banner{padding:12px 16px;border-radius:8px;font-weight:600;margin:12px 0}
.ok{background:#dafbe1;color:#116329} .bad{background:#ffebe9;color:#a40e26} .warn{background:#fff8c5;color:#7d4e00}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #d1d9e0;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f6f8fa}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-weight:600;font-size:12px}
details{margin-top:4px} pre{white-space:pre-wrap;font-size:12px;margin:4px 0}
"""


def _badge(result: dict) -> str:
    if result["status"] == "pass":
        return '<span class="badge ok">PASS</span>'
    css = "bad" if result["severity"] == "critical" else "warn"
    return f'<span class="badge {css}">FAIL</span>'


def summarize(results: list[dict]) -> dict:
    """Сводка: число правил, упавшие critical и warning."""
    failed = [r for r in results if r["status"] == "fail"]
    critical = [r["rule_id"] for r in failed if r["severity"] == "critical"]
    warnings = [r["rule_id"] for r in failed if r["severity"] == "warning"]
    return {
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed_critical": critical,
        "failed_warning": warnings,
        "status": "FAILED" if critical else ("PASSED_WITH_WARNINGS" if warnings else "PASSED"),
    }


def render_html(snapshot: dict, results: list[dict], summary: dict, monthly_counts: list[dict]) -> str:
    """Собирает самодостаточную HTML-страницу без внешних зависимостей."""
    esc = html.escape
    banner_css = {"PASSED": "ok", "PASSED_WITH_WARNINGS": "warn", "FAILED": "bad"}[summary["status"]]
    rows = []
    for r in sorted(results, key=lambda x: x["rule_id"]):
        examples = ""
        if r["examples"]:
            examples = (
                f"<details><summary>Примеры строк ({len(r['examples'])})</summary>"
                f"<pre>{esc(json.dumps(r['examples'], ensure_ascii=False, indent=1))}</pre></details>"
            )
        rows.append(
            "<tr>"
            f"<td>{esc(r['rule_id'])}</td><td>{esc(r['title'])}</td>"
            f"<td>{esc(r['severity'])}</td><td>{_badge(r)}</td>"
            f"<td>{esc(str(r['observed']))}{examples}</td><td>{esc(str(r['threshold']))}</td>"
            "</tr>"
        )
    month_rows = "".join(
        f"<tr><td>{esc(m['month'][:7])}</td><td>{m['rows']}</td></tr>" for m in monthly_counts[-24:]
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DQ отчёт {esc(snapshot['current_month'][:7])}</title><style>{STYLE}</style></head>
<body>
<h1>Data Quality: labeled_transactions</h1>
<div class="meta">Проверяемый месяц <b>{esc(snapshot['current_month'][:7])}</b>,
референс {esc(snapshot['reference_month'][:7])} · снимок {esc(snapshot['snapshot_at'])}
· отчёт {esc(datetime.now(timezone.utc).isoformat())}</div>
<div class="banner {banner_css}">Статус: {summary['status']}
{(' — упали critical: ' + ', '.join(summary['failed_critical'])) if summary['failed_critical'] else ''}
{(' — warning: ' + ', '.join(summary['failed_warning'])) if summary['failed_warning'] else ''}</div>
<div class="cards">
<div class="card">Правил<b>{summary['total']}</b></div>
<div class="card">Прошли<b>{summary['passed']}</b></div>
<div class="card">Critical fail<b>{len(summary['failed_critical'])}</b></div>
<div class="card">Warning fail<b>{len(summary['failed_warning'])}</b></div>
<div class="card">Строк в месяце<b>{snapshot['current_rows']}</b></div>
</div>
<h2>Результаты правил</h2>
<table><tr><th>ID</th><th>Правило</th><th>Severity</th><th>Статус</th><th>Факт</th><th>Порог</th></tr>
{''.join(rows)}</table>
<h2>Объём витрины по месяцам (последние 24)</h2>
<table style="width:auto"><tr><th>Месяц</th><th>Строк</th></tr>{month_rows}</table>
</body></html>"""


def build_dq_report(snapshot: dict, structure: list[dict], validity: list[dict], consistency: list[dict]) -> dict:
    """Публикует HTML- и JSON-отчёт DQ в каталог рядом с каталогом моделей.

    Читает: XCom снимка и результаты групп проверок, dq/monthly_counts.json.
    Пишет в DQ_REPORTS_DIR:
    - dq_report_<YYYY-MM>_<UTC-время>.html и .json — история отчётов;
    - latest.html и latest.json — последний отчёт (заменяются атомарно).

    Повторный запуск безопасен: создаётся ещё один отчёт с новой меткой времени,
    старые отчёты и данные не меняются, latest.* переключается целиком.
    Возвращает сводку и путь к отчёту.
    """
    results = [*structure, *validity, *consistency]
    summary = summarize(results)
    with open(snapshot["monthly_counts_path"], encoding="utf-8") as fh:
        monthly_counts = json.load(fh)

    DQ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    base = f"dq_report_{snapshot['current_month'][:7]}_{stamp}"
    page = render_html(snapshot, results, summary, monthly_counts)
    payload = {"snapshot": snapshot, "summary": summary, "results": results}

    html_path = DQ_REPORTS_DIR / f"{base}.html"
    json_path = DQ_REPORTS_DIR / f"{base}.json"
    html_path.write_text(page, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for src, name in ((html_path, "latest.html"), (json_path, "latest.json")):
        tmp = DQ_REPORTS_DIR / f".{name}.tmp"
        tmp.write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
        os.replace(tmp, DQ_REPORTS_DIR / name)
    logger.info("DQ отчёт %s: %s", html_path, summary)
    return {**summary, "report_path": str(html_path), "month": snapshot["current_month"]}
