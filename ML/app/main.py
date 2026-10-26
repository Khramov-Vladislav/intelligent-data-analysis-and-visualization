"""Taipy-приложение «Сортировка алертов SOC» (трек ML-инжиниринг).

Запуск: poetry run python app/main.py  →  http://127.0.0.1:5050
"""
import pandas as pd
from taipy.gui import Gui, State, notify

import charts
from data import CRIT_ORDER, filter_cards, kpis, load_cards, options, table_view

# Полный датасет живет на уровне процесса и не попадает в State: каждый клиент хранит только свой срез.
_cards = load_cards()
_by_id = _cards.set_index("EntityId")
_first_day, _last_day = _cards["Дата"].min(), _cards["Дата"].max()


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def fmt_hours(h: float) -> str:
    return "—" if pd.isna(h) else f"{h:,.1f} ч".replace(",", " ").replace(".", ",")


# ---- State: фильтры (меняются пользователем)
date_from = _first_day.to_pydatetime()
date_to = _last_day.to_pydatetime()
crit_lov = CRIT_ORDER
crit_sel: list[str] = []
group_lov = options(_cards, "Группа устранения")
group_sel: list[str] = []
category_lov = options(_cards, "Пиктограмма IRP")
category_sel: list[str] = []
only_open = False

# ---- State: представление (пересчитывается в callbacks)
kpi_total = kpi_fp = kpi_no_verdict = kpi_critical_open = kpi_median = ""
fig_daily = fig_verdicts = fig_crit = fig_categories = fig_hours = None
cards_table = pd.DataFrame()
period = ""

# ---- State: выбранная в таблице карточка
card_selected = False
card_title = card_alert = card_meta = ""


def _view(df: pd.DataFrame) -> dict:
    """Все производные от среза значения; одна функция для старта и для callbacks."""
    k = kpis(df)
    return {
        "kpi_total": fmt_int(k["total"]),
        "kpi_fp": f"{k['fp_share']:.1%}".replace(".", ","),
        "kpi_no_verdict": fmt_int(k["no_verdict"]),
        "kpi_critical_open": fmt_int(k["critical_open"]),
        "kpi_median": fmt_hours(k["median_hours"]),
        "fig_daily": charts.daily_dynamics(df),
        "fig_verdicts": charts.verdict_bars(df),
        "fig_crit": charts.criticality_bars(df),
        "fig_categories": charts.category_bars(df),
        "fig_hours": charts.hourly_profile(df),
        "cards_table": table_view(df),
        "period": f"{df['Создан'].min():%d.%m.%Y} — {df['Создан'].max():%d.%m.%Y} · UTC",
    }


globals().update(_view(_cards))


def _to_day(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return (ts.tz_convert(None) if ts.tzinfo else ts).normalize()


def on_filter(state: State, var_name: str = "", value=None):
    """Callback любого фильтра: пересчитывает срез и все зависимые элементы страницы."""
    start, end = _to_day(state.date_from), _to_day(state.date_to)
    if start > end:
        notify(state, "warning", "Дата начала позже даты окончания.")
        return
    df = filter_cards(
        _cards, start, end, state.crit_sel, state.group_sel, state.category_sel, state.only_open
    )
    if df.empty:
        notify(state, "warning", "Под выбранные фильтры не попала ни одна карточка — ослабьте условия.")
        return
    with state as s:  # одно обновление клиента вместо десятка
        for name, val in _view(df).items():
            setattr(s, name, val)
        s.card_selected = False


def on_reset(state: State):
    """Кнопка «Сбросить»: возвращает фильтры к значениям по умолчанию."""
    with state as s:
        s.date_from, s.date_to = _first_day.to_pydatetime(), _last_day.to_pydatetime()
        s.crit_sel, s.group_sel, s.category_sel = [], [], []
        s.only_open = False
    on_filter(state)


def on_row(state: State, var_name: str, payload: dict):
    """Клик по строке таблицы: показывает подробности карточки в боковой панели."""
    entity_id = int(state.cards_table.loc[payload["index"], "EntityId"])
    card = _by_id.loc[entity_id]
    with state as s:
        s.card_title = f"Карточка {entity_id} · {card['Критичность']} · {card['Вердикт']}"
        s.card_alert = card["Алерт"] or "Текст алерта отсутствует"
        s.card_meta = (
            f"Создана: {card['Создан']:%d.%m.%Y %H:%M} UTC  \n"
            f"Категория IRP: {card['Пиктограмма IRP'] or '—'}  \n"
            f"Группа устранения: {card['Группа устранения'] or '—'}  \n"
            f"Хост цели: {card['Hostname Цели'] or '—'}  \n"
            f"Процесс: {card['processName'] or '—'}  \n"
            f"Ответственный: {card['Ответственный'] or '—'}  \n"
            f"Время решения: {fmt_hours(card['Время решения, ч'])}"
        )
        s.card_selected = True


def on_close_card(state: State):
    state.card_selected = False


overview = """
# Сортировка алертов SOC: обзор потока карточек

<|{period}|text|class_name=subtitle|>

<|layout|columns=280px 1|gap=16px|
<|part|class_name=card filters|
### Фильтры

<|{date_from}|date|label=С|format=dd.MM.yyyy|min={_first_day}|max={_last_day}|on_change=on_filter|class_name=fullwidth|>

<|{date_to}|date|label=По|format=dd.MM.yyyy|min={_first_day}|max={_last_day}|on_change=on_filter|class_name=fullwidth|>

<|{crit_sel}|selector|lov={crit_lov}|multiple|dropdown|label=Критичность|on_change=on_filter|class_name=fullwidth|>

<|{group_sel}|selector|lov={group_lov}|multiple|dropdown|label=Группа устранения|on_change=on_filter|class_name=fullwidth|>

<|{category_sel}|selector|lov={category_lov}|multiple|dropdown|label=Категория IRP|on_change=on_filter|class_name=fullwidth|>

<|{only_open}|toggle|label=Только без вердикта|on_change=on_filter|>

<|Сбросить|button|on_action=on_reset|>

<|part|render={card_selected}|class_name=card-details|
#### <|{card_title}|text|>

<|{card_alert}|text|class_name=alert-text|>

<|{card_meta}|text|mode=markdown|>

<|Скрыть|button|on_action=on_close_card|>
|>
|>

<|
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

*Клик по строке открывает подробности карточки в панели слева.*

<|{cards_table}|table|page_size=15|width=100%|date_format=dd.MM.yyyy HH:mm|on_action=on_row|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page=overview, css_file="style.css").run(
        title="SOC · сортировка алертов", dark_mode=False, port=5050, use_reloader=False
    )
