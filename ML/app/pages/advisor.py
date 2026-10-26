"""Страница «Модель-советчик»: загрузка .joblib и асинхронный пакетный инференс (практика 3.6).

Модель загружается один раз при старте сервера и общая для всех клиентов. Инференс идет
в отдельном потоке через invoke_long_callback: интерфейс остается отзывчивым, прогресс
приходит в State периодическим статус-callback.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from taipy.gui import Markdown, State, get_state_id, invoke_long_callback, notify

import charts
from advisor import TEST_IDS, build_features, labels, load_model, load_raw, model_path, predict_scores
from data import NO_VERDICT

CHUNK = 5_000
QUEUE, HOLDOUT = "очередь", "отложенная выборка"

# ---- Инициализация модели (один раз на процесс)
try:
    _model, _meta = load_model()
    model_ready = True
    model_status = f"Загружена: {_meta['name']} · {model_path().name}"
except Exception as e:  # приложение должно работать и без модели: обзор от нее не зависит
    _model, _meta = None, {"threshold": 0.5}
    model_ready = False
    model_status = f"Модель не загружена ({model_path()}): {e}"


def _jobs() -> pd.DataFrame:
    """Что скорим: очередь без вердикта (для разбора) и отложенную выборку с фактом (для метрик)."""
    raw = load_raw()
    queue = raw[raw["Вердикт"] == NO_VERDICT].assign(Набор=QUEUE)
    # Для модели курса без списка отложенных карточек берем все размеченные (метрики будут оптимистичнее).
    labeled = raw[labels(raw).notna()]
    held = labeled[labeled["EntityId"].isin(np.load(TEST_IDS))] if TEST_IDS.exists() else labeled
    return pd.concat([queue, held.assign(Набор=HOLDOUT)], ignore_index=True)


_job_frame = _jobs() if model_ready else pd.DataFrame()
_progress: dict[str, float] = {}  # state_id → доля выполненного; пишет поток инференса, читает статус-callback


def fmt_pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


m = _meta.get("metrics_test", {})
model_card = (
    f"Порог по умолчанию: **{_meta['threshold']:.2f}** (полнота ≥ {_meta.get('min_recall', 0):.0%})  \n"
    f"Отложенная выборка: ROC-AUC {m.get('roc_auc', 0):.2f}, AP {m.get('average_precision', 0):.2f}  \n"
    f"scikit-learn при обучении: {_meta.get('sklearn', '—')}"
)
tc = _meta.get("temporal_check")
drift_note = (
    f"Проверка по времени (карточки с {pd.Timestamp(tc['test_from']):%d.%m.%Y}): ROC-AUC {tc['roc_auc']:.2f} — "
    "доля инцидентов меняется от месяца к месяцу, на новом потоке качество заметно ниже."
    if tc else ""
)

# ---- State
running = False
progress = 0
run_info = "Инференс еще не запускался."
has_result = False
threshold = float(_meta["threshold"])
k_queue = k_flagged = k_load = k_precision = k_recall = k_missed = ""
fig_cm = fig_pr = fig_hist = None
QUEUE_COLUMNS = ["EntityId", "Создан", "Критичность", "Пиктограмма IRP", "Наименование", "Оценка"]
queue_table = pd.DataFrame(columns=QUEUE_COLUMNS)  # колонки нужны сразу: по ним Taipy строит таблицу
scored = pd.DataFrame()  # оценки по всем карточкам, пишется только в on_status


def _run_inference(state_id: str) -> pd.DataFrame:
    """Выполняется в отдельном потоке: State здесь недоступен, прогресс пишем в общий словарь."""
    started = time.perf_counter()
    scores = np.empty(len(_job_frame))
    parts = np.array_split(np.arange(len(_job_frame)), max(len(_job_frame) // CHUNK, 1))
    for i, idx in enumerate(parts):
        X = build_features(_job_frame.iloc[idx])
        scores[idx] = predict_scores(_model, X)
        _progress[state_id] = (i + 1) / len(parts)
    out = _job_frame[["EntityId", "Создан", "Критичность", "Пиктограмма IRP", "Наименование", "Набор"]].copy()
    out["Оценка"] = scores
    out["y"] = labels(_job_frame).to_numpy()
    out.attrs["seconds"] = time.perf_counter() - started
    return out


def _apply_threshold(state: State):
    """Пересчет всего, что зависит от порога: дешево, поэтому выполняется синхронно."""
    res, t = state.scored, state.threshold
    queue = res[res["Набор"] == QUEUE]
    held = res[res["Набор"] == HOLDOUT]
    y, p = held["y"].to_numpy(dtype=int), held["Оценка"].to_numpy()
    pred = p >= t
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    flagged = int((queue["Оценка"] >= t).sum())
    curve_p, curve_r, _ = precision_recall_curve(y, p)

    view = queue.sort_values("Оценка", ascending=False).head(500).copy()
    view["Оценка"] = view["Оценка"].round(3)
    view["Критичность"] = view["Критичность"].astype(str)
    view["Создан"] = pd.to_datetime(view["Создан"]).dt.strftime("%d.%m.%Y %H:%M")  # UTC

    with state as s:
        s.k_queue = fmt_int(len(queue))
        s.k_flagged = fmt_int(flagged)
        s.k_load = fmt_pct(flagged / max(len(queue), 1))
        s.k_precision = fmt_pct(precision)
        s.k_recall = fmt_pct(recall)
        s.k_missed = fmt_int(fn)
        s.fig_cm = charts.confusion_matrix_fig(tp, fn, fp, tn)
        s.fig_pr = charts.pr_curve_fig(curve_p, curve_r, precision, recall)
        s.fig_hist = charts.score_histogram(queue["Оценка"].to_numpy(), t)
        s.queue_table = view[QUEUE_COLUMNS]


def on_run(state: State):
    """Кнопка «Запустить инференс»: стартует фоновый поток и сразу возвращает управление UI."""
    if not model_ready:
        notify(state, "error", "Модель не загружена — инференс невозможен.")
        return
    sid = get_state_id(state)
    _progress[sid] = 0.0
    with state as s:
        s.running, s.progress = True, 0
        s.run_info = f"Идет инференс {fmt_int(len(_job_frame))} карточек…"
    invoke_long_callback(state, _run_inference, [sid], on_status, [], period=500)


def on_status(state: State, status, result=None):
    """Периодически (int) — обновляет прогресс; в конце (bool) — принимает результат или сообщает об ошибке."""
    sid = get_state_id(state)
    if isinstance(status, bool):
        _progress.pop(sid, None)
        state.running = False
        if not status:
            state.run_info = "Инференс завершился с ошибкой — подробности в журнале сервера."
            notify(state, "error", "Инференс завершился с ошибкой.")
            return
        with state as s:
            s.scored = result
            s.progress = 100
            s.has_result = True
            s.run_info = (f"Готово: {fmt_int(len(result))} карточек за {result.attrs['seconds']:.1f} с "
                          f"(очередь и отложенная выборка).")
        _apply_threshold(state)
        notify(state, "success", "Инференс завершен.")
    else:
        state.progress = int(100 * _progress.get(sid, 0))


def on_threshold(state: State, var_name: str = "", value=None):
    if state.has_result:
        _apply_threshold(state)


def on_reset_threshold(state: State):
    state.threshold = float(_meta["threshold"])
    on_threshold(state)


page = Markdown("""
# Модель-советчик: очередь карточек без вердикта

<|{model_status}|text|class_name=subtitle|>

<|layout|columns=320px 1|gap=16px|
<|part|class_name=card filters|
### Инференс

<|{model_card}|text|mode=markdown|>

<|{drift_note}|text|class_name=note|>

<|Запустить инференс|button|on_action=on_run|active={model_ready and not running}|>

<|{progress}|progress|show_value|linear|render={running or has_result}|>

<|{run_info}|text|class_name=note|>

<|part|render={has_result}|
### Порог решения

<|{threshold}|slider|min=0.01|max=0.99|step=0.01|on_change=on_threshold|continuous=False|>

Текущий порог τ = <|{threshold}|text|format=%.2f|>

<|По умолчанию|button|on_action=on_reset_threshold|>

Оценка ≥ τ → карточка уходит аналитику «к разбору» в первую очередь. Ниже порога — кандидат в FP, но итоговый вердикт все равно за аналитиком.
|>
|>

<|
<|part|render={not has_result}|class_name=card|
Запустите инференс: модель оценит карточки без вердикта и отложенную выборку с известным фактом. Пока он идет, страницей «Обзор» можно пользоваться — интерфейс не блокируется.
|>

<|part|render={has_result}|
<|layout|columns=1 1 1 1 1 1|gap=12px|
<|part|class_name=card|
**В очереди**
<|{k_queue}|text|class_name=kpi|>
|>
<|part|class_name=card|
**К разбору (≥ τ)**
<|{k_flagged}|text|class_name=kpi signal|>
|>
<|part|class_name=card|
**Доля очереди**
<|{k_load}|text|class_name=kpi|>
|>
<|part|class_name=card|
**Точность**
<|{k_precision}|text|class_name=kpi|>
|>
<|part|class_name=card|
**Полнота**
<|{k_recall}|text|class_name=kpi|>
|>
<|part|class_name=card|
**Пропущено (FN)**
<|{k_missed}|text|class_name=kpi signal|>
|>
|>

*Точность, полнота и FN — на отложенной выборке с известным фактом; доля очереди — на карточках без вердикта.*

<|layout|columns=1 1|gap=12px|
<|chart|figure={fig_cm}|>

<|chart|figure={fig_pr}|>
|>

<|chart|figure={fig_hist}|>

## Очередь по убыванию оценки (топ-500)

<|{queue_table}|table|page_size=15|width=100%|>
|>
|>
|>
""")
