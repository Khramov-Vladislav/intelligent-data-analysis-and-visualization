"""Вкладка «Почему так?» для модели курса."""
import pandas as pd
from taipy.gui import Markdown, State

import charts
from course_model import COURSE_VERSION, configured_course
from course_shap import CONTRIB_COLUMNS, TEAM_COLUMNS, CourseShap, display_value

_kit, _model, _error = configured_course()
try:
    if _error is not None:
        raise _error
    _shap = CourseShap(_kit, _model)
except Exception as error:
    _shap = None
    course_xai_status = f"Не удалось открыть объяснения модели курса. {error}"

course_xai_ready = _shap is not None
team_sel = ""
team_options = []
team_cards = pd.DataFrame(columns=TEAM_COLUMNS)
course_contributions = pd.DataFrame(columns=CONTRIB_COLUMNS)
course_waterfall = course_top = course_importance = None
course_case = course_context = course_comparison_note = ""

if course_xai_ready:
    team_options = [str(number) for number in sorted(_shap.cards["team"].unique())]
    course_xai_status = (
        f"Модель {COURSE_VERSION} · порог {_shap.threshold:.6f} · "
        f"{len(_shap.cards)} учебных карточек".replace(".", ",")
    )
    if _shap.test_importance.empty:
        course_importance = charts.shap_importance(_shap.values.to_numpy(), _shap.values.columns, top=5)
        course_comparison_note = "Для сравнения с тестом положите архив IAVD_3.7_BI_team.zip рядом с комплектом ML."
    else:
        course_importance = charts.shap_importance_comparison(_shap.practice_importance, _shap.test_importance)
        course_comparison_note = (
            "Важность – средний модуль вклада по карточкам. "
            "Наборы практики собраны с равным числом исходов TP, TN, FN и FP. "
            "Из-за разного состава выборок порядок признаков может меняться."
        )


def _show_card(state: State, entity_id: int):
    row = _shap.cards.loc[entity_id]
    raw = _shap.raw.loc[entity_id]
    contributions = _shap.contributions(entity_id)
    probability = _shap.probability.loc[entity_id]
    base = _shap.base.loc[entity_id]
    log_odds = _shap.log_odds.loc[entity_id]
    decision = "инцидент" if probability >= _shap.threshold else "ниже порога"
    with state as s:
        s.course_case = (
            f"**Карточка {entity_id} · {row['outcome']}**  \n"
            f"Вердикт аналитика – {raw['Вердикт']}. Решение модели – {decision}.  \n"
            f"Оценка – {probability:.6f} при пороге {_shap.threshold:.6f}.  \n"
            f"База ({base:+.4f}) + вклады ({log_odds - base:+.4f}) = итог ({log_odds:+.4f}).  \n"
            f"Расхождение с выходом модели до перевода в вероятность – {_shap.raw_error.loc[entity_id]:.2e}.  \n"
            f"Расхождение вероятности с CSV – {_shap.csv_error.loc[entity_id]:.2e}."
        )
        s.course_context = (
            f"**Правило SIEM** – {display_value(raw['Наименование'])}  \n"
            f"**Критичность** – {display_value(raw['Критичность'])}  \n"
            f"**Алерт** – {display_value(raw['Алерт'], limit=250)}"
        )
        s.course_contributions = contributions.head(10).assign(Вклад=lambda top: top["Вклад"].round(4))
        s.course_waterfall = charts.shap_waterfall(contributions, base, title="База, вклады и итог")
        s.course_top = charts.shap_top_bars(contributions)


def on_team(state: State, var_name: str, value: str):
    if not value:
        state.team_cards = pd.DataFrame(columns=TEAM_COLUMNS)
        return
    cards = _shap.team(int(value))
    state.team_cards = cards
    first_fn = cards.loc[cards["Исход"].eq("FN"), "EntityId"].iloc[0]
    _show_card(state, int(first_fn))


def on_team_card(state: State, var_name: str, payload: dict):
    _show_card(state, int(state.team_cards.loc[payload["index"], "EntityId"]))


page = Markdown("""
# Почему так?

<|{course_xai_status}|text|class_name=subtitle|>

<|part|render={course_xai_ready}|
<|{team_sel}|selector|lov={team_options}|dropdown|label=Номер набора|on_change=on_team|>

<|part|render={not team_sel}|
Выберите согласованный набор. Любой номер можно открыть для проверки интерфейса.
|>

<|part|render={team_sel != ""}|
## Карточки набора

Выберите строку, чтобы посмотреть, какие признаки повлияли на оценку.

<|{team_cards}|table|page_size=8|width=100%|on_action=on_team_card|>

<|layout|columns=1 2|gap=16px|
<|part|class_name=card|
<|{course_case}|text|mode=markdown|>

<|{course_context}|text|mode=markdown|>
|>

<|
<|chart|figure={course_waterfall}|>
|>
|>

Красный вклад повышает оценку инцидента, синий снижает. Водопад учитывает сумму всех признаков.

<|chart|figure={course_top}|>

<|{course_contributions}|table|page_size=10|width=100%|>
|>

## Важность на выборке

<|chart|figure={course_importance}|>

<|{course_comparison_note}|text|class_name=note|>
|>
""")
