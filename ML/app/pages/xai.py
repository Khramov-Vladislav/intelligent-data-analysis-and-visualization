"""Страница «Объяснения (XAI)»: глобальная важность по предрасчитанным SHAP и разбор учебных карточек (практика 3.7)."""
import numpy as np
import pandas as pd
from taipy.gui import Markdown, State, invoke_long_callback, notify

import charts
from advisor import NUMERIC, SHAP_SAMPLE, build_features, load_raw, partial_dependence, pdp_grid
from explain import card_explanation
from pages.advisor import _meta, _model, model_ready

_tau = float(_meta["threshold"])
shap_ready = model_ready and SHAP_SAMPLE.exists()
shap_status = (
    "Глобальные графики построены по предрасчитанным SHAP-значениям отложенной выборки."
    if shap_ready else f"Нет файла {SHAP_SAMPLE.name} — запустите model/train_baseline.py."
)


def _examples(sample) -> dict[str, int]:
    """Учебные карточки из выборки: самые показательные исходы каждого типа на рабочем пороге."""
    df = pd.DataFrame({"id": sample["entity_id"], "y": sample["y"], "p": sample["score"]})
    picks = {
        "TP — найденный инцидент (самая уверенная тревога)": df[df.y == 1].nlargest(1, "p"),
        "FN — пропущенный инцидент (самая низкая оценка)": df[(df.y == 1) & (df.p < _tau)].nsmallest(1, "p"),
        "FN — пропущенный инцидент рядом с порогом": df[(df.y == 1) & (df.p < _tau)].nlargest(1, "p"),
        "FP — ложная тревога (самая высокая оценка)": df[df.y == 0].nlargest(1, "p"),
        "TN — корректное молчание": df[df.y == 0].nsmallest(1, "p"),
    }
    return {name: int(rows["id"].iloc[0]) for name, rows in picks.items() if len(rows)}


if shap_ready:
    _sample = np.load(SHAP_SAMPLE, allow_pickle=False)
    _values, _data, _features = _sample["values"], _sample["data"], list(_sample["features"])
    _example_ids = _examples(_sample)
    fig_importance = charts.shap_importance(_values, _features)
    _top = pd.Series(np.abs(_values).mean(axis=0), index=_features).sort_values(ascending=False)
    feature_lov = _top.index.tolist()
    feature_sel = feature_lov[0]
    fig_by_value = charts.shap_by_value(_values, _data, feature_sel, _features)
else:
    _example_ids, feature_lov, feature_sel = {}, [], ""
    fig_importance = fig_by_value = None

PDP_CARDS = 1000


def _pdp_sample() -> pd.DataFrame:
    """Признаки 1 000 карточек отложенной выборки: на них считаются PDP и ICE."""
    ids = _sample["entity_id"][:PDP_CARDS]
    raw = load_raw()
    return build_features(raw[raw["EntityId"].isin(ids)]).reset_index(drop=True)


def _pdp(feature: str):
    grid = pdp_grid(_X_pdp, feature)
    pdp, ice = partial_dependence(_model, _X_pdp, feature, grid)
    return charts.pdp_fig(feature, grid, pdp, ice, _X_pdp[feature], _tau, numeric=feature in NUMERIC)


if shap_ready:
    _X_pdp = _pdp_sample()
    pdp_feature = "Час"
    fig_pdp = _pdp(pdp_feature)
else:
    pdp_feature, fig_pdp = "", None
pdp_busy = False

example_lov = list(_example_ids)
example_sel = example_lov[1] if len(example_lov) > 1 else (example_lov[0] if example_lov else "")
entity_input = _example_ids.get(example_sel, 0)
fig_local, local_info = card_explanation(_model, entity_input, _tau) if entity_input else (None, "")


def on_feature(state: State, var_name: str, value):
    state.fig_by_value = charts.shap_by_value(_values, _data, value, _features)


def on_pdp_feature(state: State, var_name: str, value):
    """PDP пересчитывает модель 10–25 раз на 1 000 карточках (до ~1 с) — считаем в фоне, UI не ждет."""
    state.pdp_busy = True
    invoke_long_callback(state, _pdp, [value], on_pdp_done)


def on_pdp_done(state: State, status, fig=None):
    state.pdp_busy = False
    if status is True:
        state.fig_pdp = fig
    else:
        notify(state, "error", "Не удалось построить PDP — подробности в журнале сервера.")


def _show(state: State, entity_id: int):
    res = card_explanation(_model, int(entity_id), _tau)
    if res is None:
        notify(state, "warning", f"Карточки {entity_id} нет в датасете.")
        return
    with state as s:
        s.fig_local, s.local_info = res
        s.entity_input = int(entity_id)


def on_example(state: State, var_name: str, value):
    _show(state, _example_ids[value])


def on_explain_id(state: State):
    _show(state, state.entity_input)


page = Markdown("""
# Объяснения модели (XAI)

<|{shap_status}|text|class_name=subtitle|>

<|layout|columns=1 1|gap=16px|
<|chart|figure={fig_importance}|>

<|
<|{feature_sel}|selector|lov={feature_lov}|dropdown|label=Признак|on_change=on_feature|class_name=fullwidth|>

<|chart|figure={fig_by_value}|>
|>
|>

## Частичная зависимость (PDP)

<|layout|columns=320px 1|gap=16px|
<|part|class_name=card filters|
<|{pdp_feature}|selector|lov={feature_lov}|dropdown|label=Признак|on_change=on_pdp_feature|active={not pdp_busy}|class_name=fullwidth|>

<|part|render={pdp_busy}|
<|Считаю PDP…|text|class_name=note|>
|>

PDP показывает, как в среднем меняется оценка модели, если всем 1 000 карточкам выборки подставить одно
значение признака. Серые линии ICE — то же для отдельных карточек: если они идут по-разному, среднее
скрывает взаимодействие признаков.

Ограничения (лекция 3.3): подстановка создает сочетания, которых может не быть в данных (например, ночной час
у дневного правила), — поэтому сетка числовых признаков ограничена 5–95 % наблюдаемых значений, а снизу
показано, сколько реальных карточек приходится на каждое значение. При коррелированных признаках PDP
приписывает эффект одному признаку, хотя модель могла опираться на связанный.
|>

<|chart|figure={fig_pdp}|>
|>

## Разбор карточки

<|layout|columns=320px 1|gap=16px|
<|part|class_name=card filters|
<|{example_sel}|selector|lov={example_lov}|label=Учебные карточки|on_change=on_example|class_name=fullwidth|>

<|{entity_input}|number|label=Или EntityId любой карточки|>

<|Объяснить|button|on_action=on_explain_id|>

<|{local_info}|text|mode=markdown|>
|>

<|
<|chart|figure={fig_local}|>

*Как читать: ось — лог-шансы класса «инцидент». Модель начинает со средней оценки по обучающей выборке; красные полосы
сдвигают оценку к «инциденту», синие — к «ложному срабатыванию»; итог переводится в оценку карточки (сигмоида).
Вклад признака зависит от остальных значений карточки — это объяснение модели, а не причинность.*
|>
|>
""")
