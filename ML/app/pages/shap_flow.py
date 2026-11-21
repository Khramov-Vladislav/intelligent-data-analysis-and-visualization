"""Проектный разбор SHAP на потоке эксплуатации SOC."""
import os
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from taipy.gui import Markdown, State, notify

import charts
from course_model import COURSE_DATA, COURSE_VERSION, configured_course
from shap_flow import CATEGORY, OUTCOMES, ShapFlow

_kit, _, _ = configured_course()
_flow = None
if _kit is None:
    flow_status = "Для просмотра потока укажите путь к комплекту ML 3.7 в IAVD_COURSE_KIT."
else:
    try:
        _flow = ShapFlow(
            _kit.parent / "IAVD_3.7_BI_team.zip",
            Path(os.environ.get("IAVD_COURSE_DATA", COURSE_DATA)).expanduser(),
        )
        c = _flow.cards
        flow_status = f"{COURSE_VERSION} · {len(c):,} карточки потока · {c['y'].notna().sum():,} с вердиктом".replace(",", " ")
    except FileNotFoundError:
        flow_status = "Нужны IAVD_3.7_BI_team.zip рядом с комплектом ML 3.7 и iavd_ml_dataset.parquet."
    except (BadZipFile, ImportError, OSError, KeyError, ValueError) as error:
        flow_status = f"Не удалось открыть данные SHAP. {error}"

flow_ready = _flow is not None
flow_missed = flow_check = flow_tau = flow_group_note = ""
flow_importance = flow_source = flow_grouping = flow_categories = flow_outcomes = flow_waterfall = None
flow_category_options = []
flow_category = ""
flow_id = 0
flow_case = flow_context = ""
flow_contrib = pd.DataFrame()


def decimal(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def show_group(category: str) -> tuple[object, str, str]:
    cards = _flow.cards.loc[_flow.cards[CATEGORY].eq(category)]
    mask = cards["outcome"].eq("FN")
    missed = int(mask.sum())
    incidents = int(cards["y"].eq(1).sum())
    found = incidents - missed
    fig = charts.shap_outcome_fig(_flow.by_outcome(cards))
    note = f"Модель пропустила {missed} из {incidents} инцидентов ({decimal(100 * missed / incidents, 1)} %)."
    if not missed or not found:
        note += " В этой категории представлен один из двух исходов."
    # Если есть пропуски, выбираем ближайший к порогу; иначе – найденный инцидент.
    recommended = (cards.loc[mask] if missed else cards.loc[cards["outcome"].eq("TP")])["score"].idxmax()
    return fig, note, recommended


def show_card(card_id: str) -> tuple:
    card = _flow.cards.loc[card_id]
    contributions = _flow.contributions(card_id)
    waterfall = charts.shap_waterfall(
        contributions, card["base"], title="База, вклады и итог", logit_threshold=_flow.logit_threshold,
    )
    outcome = OUTCOMES.get(card["outcome"], "Нет метки")
    case = (
        f"**Карточка {card_id} · {outcome}**  \n"
        f"Оценка модели – {decimal(card['score'], 6)} при пороге {flow_tau}.  \n"
        f"База {decimal(card['base'], 4)} + вклады {decimal(card['z'] - card['base'], 4)} "
        f"= итог {decimal(card['z'], 4)} в логарифме шансов."
    )
    context = (
        f"**Вердикт аналитика** – {card['verdict'] or 'не указан'}  \n"
        f"**Категория IRP** – {card[CATEGORY]}  \n"
        f"**Правило SIEM** – {card['rule']}  \n"
        f"**Адрес источника** – {card['Адрес источника'].lower()}  \n"
        f"**Поле группировки** – {card['Поле группировки'].lower()}"
    )
    table = contributions.head(10).assign(Вклад=lambda top: top["Вклад"].map(lambda value: decimal(value, 4)))
    return waterfall, case, context, table


if flow_ready:
    _cards = _flow.cards
    _misses = _flow.misses_by_category().sort_values(["missed", "incidents"], ascending=False)
    flow_category_options = _misses.index.tolist()
    flow_category = flow_category_options[0]
    _positives, _fn = int(_misses["incidents"].sum()), int(_misses["missed"].sum())
    _positive_count = f"{_positives:,}".replace(",", " ")
    flow_missed = f"{_fn} из {_positive_count} ({decimal(100 * _fn / _positives, 1)} %)"
    flow_check = f"{_cards['error'].max():.2e}"
    flow_tau = decimal(_flow.threshold, 6)
    flow_importance = charts.shap_importance(_flow.values.to_numpy(), _flow.values.columns, top=8)
    flow_importance.update_yaxes(automargin=True)
    flow_source = charts.shap_evidence_fig(_flow.evidence(_cards, "Адрес источника"))
    flow_grouping = charts.shap_evidence_fig(_flow.evidence(_cards, "Поле группировки"))
    flow_categories = charts.missed_by_segment_fig(_misses)
    flow_outcomes, flow_group_note, _default_card = show_group(flow_category)
    flow_id = int(_default_card)
    flow_waterfall, flow_case, flow_context, flow_contrib = show_card(_default_card)


def on_flow_category(state: State, var_name: str, value: str):
    fig, note, card_id = show_group(value)
    with state as s:
        s.flow_outcomes, s.flow_group_note = fig, note
        s.flow_id = int(card_id)
        s.flow_waterfall, s.flow_case, s.flow_context, s.flow_contrib = show_card(card_id)


def on_flow_card(state: State):
    card_id = str(int(state.flow_id)) if state.flow_id and int(state.flow_id) == state.flow_id else ""
    if card_id not in _flow.cards.index:
        notify(state, "warning", "Карточки с таким ID нет в срезе эксплуатации.")
        return
    with state as s:
        s.flow_waterfall, s.flow_case, s.flow_context, s.flow_contrib = show_card(card_id)


page = Markdown("""
# Объяснения на потоке

<|{flow_status}|text|class_name=subtitle|>

<|part|render={flow_ready}|
<|part|class_name=inference-kpis|
<|part|class_name=card|
**Пропущено инцидентов**

<|{flow_missed}|text|class_name=kpi signal|>
|>
<|part|class_name=card|
**Порог модели**

<|{flow_tau}|text|class_name=kpi|>
|>
<|part|class_name=card|
**Максимальная разница с оценкой модели**

<|{flow_check}|text|class_name=kpi|>
|>
|>

<|part|class_name=card inference-chart flow-section flow-overview|
## 1. Что влияет на оценку в потоке

<|chart|figure={flow_importance}|>

<|part|class_name=inference-panels|
<|part|class_name=inference-chart|
### Адрес источника

<|chart|figure={flow_source}|>

Положительный вклад внешнего адреса согласуется с более высокой долей инцидентов.
|>
<|part|class_name=inference-chart|
### Поле группировки

<|chart|figure={flow_grouping}|>

У пустого поля вклад выше, хотя доли инцидентов почти одинаковы.
|>
|>

*Средние вклады и доли в двух сравнениях рассчитаны по карточкам с вердиктом.*
|>

<|part|class_name=card inference-chart flow-section flow-groups|
## 2. Где модель пропускает инциденты

<|{flow_category}|selector|lov={flow_category_options}|dropdown|label=Категория IRP|on_change=on_flow_category|>

<|{flow_group_note}|text|class_name=note|>

<|part|class_name=inference-panels|
<|part|class_name=inference-chart|
### Пропуски по категориям

<|chart|figure={flow_categories}|>
|>
<|part|class_name=inference-chart|
### Вклады у пропущенных и найденных

<|chart|figure={flow_outcomes}|>
|>
|>
|>

<|part|class_name=card inference-chart flow-section flow-card|
## 3. Почему модель так решила по карточке

<|{flow_id}|number|label=ID карточки|step=1|min=1|>
<|Открыть карточку|button|on_action=on_flow_card|>

<|part|class_name=flow-card-grid|
<|part|
<|{flow_case}|text|mode=markdown|>
<|{flow_context}|text|mode=markdown|>
|>
<|part|
<|chart|figure={flow_waterfall}|>
|>
|>

<|part|class_name=inference-table|
<|{flow_contrib}|table|show_all|rebuild|>
|>
|>
|>
""")
