"""Качество сохранённых прогнозов на учебном логе инференса."""
from math import isfinite

import pandas as pd
from taipy.gui import Markdown, State, notify

import charts
from inference_log import LOG_THRESHOLD, SEGMENTS, SRS_THRESHOLD, load_log, log_path
from inference_metrics import incidents_by, threshold_counts

_log_path = log_path()
try:
    _log = load_log(_log_path)
    log_status = f"Модель {_log['version'].iloc[0]} · учебный лог прогнозов"
except FileNotFoundError:
    _log = pd.DataFrame()
    log_status = "Учебный лог не найден. Укажите путь к MODEL_PREDICTIONS_target.parquet в IAVD_INFERENCE_LOG."
except (OSError, ValueError, KeyError) as error:
    _log = pd.DataFrame()
    log_status = f"Не удалось открыть учебный лог. {error}"

log_ready = not _log.empty
log_first = _log["Дата"].min().to_pydatetime() if log_ready else None
log_last = _log["Дата"].max().to_pydatetime() if log_ready else None
log_from, log_to = log_first, log_last
log_threshold = LOG_THRESHOLD
log_segment_options = list(SEGMENTS)
log_segment = log_segment_options[0]


def count_text(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f} %".replace(".", ",") if denominator else "–"


def as_day(value) -> pd.Timestamp:
    day = pd.Timestamp(value)
    return pd.NaT if pd.isna(day) else day.tz_localize(None).normalize()


def _view(start, end, threshold: float, segment: str) -> dict:
    start, end = as_day(start), as_day(end)
    frame = _log.loc[_log["Дата"].between(start, end)] if log_ready else _log
    total = len(frame)
    coverage = int(frame["y"].notna().sum()) if total else 0
    values = {
        "has_predictions": total > 0,
        "has_incidents": False,
        "coverage_text": f"Размечено {count_text(coverage)} из {count_text(total)} прогнозов.",
        "kpi_miss": "–",
        "miss_detail": "",
        "kpi_precision": "–",
        "precision_detail": "",
        "kpi_queue": "–",
        "queue_detail": "",
        "weekly_fig": None,
        "segment_fig": None,
        "tradeoff_fig": None,
        "comparison": pd.DataFrame(columns=["Показатель", "Порог лога 0,18", "Выбранный порог", "Разница"]),
    }
    if pd.isna(start) or pd.isna(end):
        values["coverage_text"] = "Укажите обе даты периода."
    elif start > end:
        values["coverage_text"] = "Дата начала позже даты окончания."
    elif total == 0:
        values["coverage_text"] = "За выбранный период прогнозов нет."
    if total == 0:
        return values

    scores, labels = frame["score"], frame["y"]
    curve = threshold_counts(scores, labels)
    selected = curve.loc[curve.index >= threshold].iloc[-1]
    reference = curve.loc[curve.index >= LOG_THRESHOLD].iloc[-1]
    tp, fp, fn, flagged = (int(selected[column]) for column in ("TP", "FP", "FN", "flagged"))
    positives = tp + fn
    difference = []
    for title, key, available in (
        ("К разбору", "flagged", True),
        ("Пропущено инцидентов", "FN", positives > 0),
        ("Ложных тревог среди размеченных", "FP", coverage > 0),
    ):
        before, after = int(reference[key]), int(selected[key])
        delta = after - before
        difference.append((
            title, count_text(before) if available else "–", count_text(after) if available else "–",
            f"{'+' if delta > 0 else '−' if delta < 0 else ''}{count_text(abs(delta))}" if available else "–",
        ))

    values.update({
        "has_incidents": positives > 0,
        "kpi_miss": percent(fn, positives),
        "miss_detail": f"{count_text(fn)} из {count_text(positives)} инцидентов" if positives else "Нет размеченных инцидентов",
        "kpi_precision": percent(tp, tp + fp),
        "precision_detail": f"{count_text(tp)} из {count_text(tp + fp)} размеченных тревог"
        if tp + fp else "Нет размеченных тревог",
        "kpi_queue": percent(flagged, total),
        "queue_detail": f"{count_text(flagged)} из {count_text(total)} прогнозов",
        "comparison": pd.DataFrame(difference, columns=values["comparison"].columns),
    })
    if positives:
        missed = scores.lt(threshold)
        weeks = incidents_by(labels, missed, frame["Неделя"])
        segments = incidents_by(labels, missed, frame[SEGMENTS[segment]])
        values["weekly_fig"] = charts.missed_by_week_fig(weeks)
        values["segment_fig"] = charts.missed_by_segment_fig(segments.loc[segments["incidents"].gt(0)])
        values["tradeoff_fig"] = charts.inference_tradeoff_fig(curve, selected, reference, total, positives)
    return values


globals().update(_view(log_from, log_to, log_threshold, log_segment))


def on_log_filter(state: State, var_name: str = "", value=None):
    threshold = state.log_threshold
    if threshold is None or not isfinite(threshold) or not 0 <= threshold <= 1:
        notify(state, "warning", "Укажите порог от 0 до 1.")
        return
    with state as s:
        for name, result in _view(s.log_from, s.log_to, threshold, s.log_segment).items():
            setattr(s, name, result)


def use_log_threshold(state: State):
    state.log_threshold = LOG_THRESHOLD
    on_log_filter(state)


def use_srs_threshold(state: State):
    state.log_threshold = SRS_THRESHOLD
    on_log_filter(state)


page = Markdown("""
# Качество модели

<|{log_status}|text|class_name=subtitle|>

<|part|render={log_ready}|
<|part|class_name=card inference-controls|
<|{log_from}|date|label=С|format=dd.MM.yyyy|min={log_first}|max={log_last}|on_change=on_log_filter|>
<|{log_to}|date|label=По|format=dd.MM.yyyy|min={log_first}|max={log_last}|on_change=on_log_filter|>
<|{log_threshold}|number|label=Порог|step=0.00001|min=0|max=1|on_change=on_log_filter|>
<|Порог лога · 0,18|button|on_action=use_log_threshold|>
<|Порог из СРС 3.2 · 0,00498|button|on_action=use_srs_threshold|>
|>

<|{coverage_text}|text|class_name=note|>

<|part|render={has_predictions}|
<|part|class_name=inference-kpis|
<|part|class_name=card|
**Пропущено инцидентов**

<|{kpi_miss}|text|class_name=kpi signal|>
<|{miss_detail}|text|class_name=note|>
|>

<|part|class_name=card|
**Тревог по делу**

<|{kpi_precision}|text|class_name=kpi|>
<|{precision_detail}|text|class_name=note|>
|>

<|part|class_name=card|
**К разбору**

<|{kpi_queue}|text|class_name=kpi|>
<|{queue_detail}|text|class_name=note|>
|>
|>

<|part|render={has_incidents}|class_name=inference-panels|
<|part|class_name=card inference-chart|
## Пропуски по неделям

Неделя начинается в понедельник.

<|chart|figure={weekly_fig}|>
|>

<|part|class_name=card inference-chart|
## В каких группах больше пропусков

<|{log_segment}|selector|lov={log_segment_options}|dropdown|label=Группировать по|on_change=on_log_filter|>

<|chart|figure={segment_fig}|>
|>
|>

<|part|class_name=card inference-comparison|
## Как порог влияет на нагрузку

К разбору идут все карточки с оценкой от порога. Ошибки считаются по карточкам с вердиктом.

<|part|render={has_incidents}|
<|chart|figure={tradeoff_fig}|>
|>

<|part|class_name=inference-table|
<|{comparison}|table|show_all|rebuild|>
|>
|>
|>
|>
""")
