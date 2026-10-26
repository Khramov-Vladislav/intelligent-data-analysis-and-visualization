"""Локальное объяснение одной карточки: общий код страниц «Модель-советчик» и «Объяснения»."""
import pandas as pd
import plotly.graph_objects as go

import charts
from advisor import SEP, build_features, explain_card, load_raw, predict_scores


def _text(value, limit: int = 300) -> str:
    if not isinstance(value, str) or not value:
        return "—"
    first = value.split(SEP)[0]
    return first if len(first) <= limit else first[: limit - 1] + "…"


def card_explanation(model, entity_id: int, threshold: float) -> tuple[go.Figure, str] | None:
    """Waterfall SHAP и сводка по карточке; None, если такой карточки нет в датасете."""
    raw = load_raw()
    row = raw[raw["EntityId"] == entity_id]
    if row.empty:
        return None
    X = build_features(row)
    contrib, base = explain_card(model, X)
    score = float(predict_scores(model, X)[0])
    r = row.iloc[0]
    decision = "к разбору" if score >= threshold else "кандидат в ложные срабатывания"
    score_line = f"Оценка модели: **{score:.1%}** → {decision} (τ = {threshold:.2f})".replace(".", ",")
    info = (
        f"**Карточка {entity_id}** · {r['Критичность']} · вердикт аналитика: {r['Вердикт']}  \n"
        f"{score_line}  \n"
        f"Создана: {pd.Timestamp(r['Создан']):%d.%m.%Y %H:%M} UTC · правило: {_text(r['Наименование'], 80)}  \n"
        f"Алерт: {_text(r['Алерт'])}"
    )
    return charts.shap_waterfall(contrib, base), info
