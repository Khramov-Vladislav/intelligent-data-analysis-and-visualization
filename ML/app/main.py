"""Taipy-приложение «Сортировка алертов SOC» (трек ML-инжиниринг).

Запуск: poetry run python app/main.py  →  http://127.0.0.1:5050
"""
from taipy.gui import Gui

import charts
from data import kpis, load_cards, table_view

_cards = load_cards()


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


_k = kpis(_cards)
kpi_total = fmt_int(_k["total"])
kpi_fp = f"{_k['fp_share']:.1%}".replace(".", ",")
kpi_no_verdict = fmt_int(_k["no_verdict"])
kpi_critical_open = fmt_int(_k["critical_open"])
kpi_median = f"{_k['median_hours']:.1f} ч".replace(".", ",")

fig_daily = charts.daily_dynamics(_cards)
fig_verdicts = charts.verdict_bars(_cards)
fig_crit = charts.criticality_bars(_cards)
fig_categories = charts.category_bars(_cards)
fig_hours = charts.hourly_profile(_cards)

cards_table = table_view(_cards)
period = f"{_cards['Создан'].min():%d.%m.%Y} — {_cards['Создан'].max():%d.%m.%Y}"

overview = """
# Сортировка алертов SOC: обзор потока карточек

<|{period}|text|class_name=subtitle|>

<|layout|columns=1 1 1 1 1|gap=12px|
<|part|class_name=card|
**Карточек**
<|{kpi_total}|text|class_name=kpi|>
|>
<|part|class_name=card|
**Ложные срабатывания**
<|{kpi_fp}|text|class_name=kpi signal|>
|>
<|part|class_name=card|
**Без вердикта**
<|{kpi_no_verdict}|text|class_name=kpi|>
|>
<|part|class_name=card|
**Критические без вердикта**
<|{kpi_critical_open}|text|class_name=kpi signal|>
|>
<|part|class_name=card|
**Медиана времени решения**
<|{kpi_median}|text|class_name=kpi|>
|>
|>

<|chart|figure={fig_daily}|>

<|layout|columns=1 1|gap=12px|
<|chart|figure={fig_verdicts}|>

<|chart|figure={fig_crit}|>

<|chart|figure={fig_categories}|>

<|chart|figure={fig_hours}|>
|>

## Карточки

<|{cards_table}|table|page_size=15|width=100%|date_format=dd.MM.yyyy HH:mm|>
"""

if __name__ == "__main__":
    Gui(page=overview, css_file="style.css").run(title="SOC · сортировка алертов", dark_mode=False, port=5050, use_reloader=False)
