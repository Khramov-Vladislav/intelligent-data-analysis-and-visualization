"""Графики Plotly для дашборда SOC.

Правила оформления (лекции 2.1–2.3): одна нейтральная серая палитра, насыщенный цвет
несет один смысл; красный — только «требует реакции» (критичность, ложные тревоги как нагрузка);
сетка и рамки сведены к минимуму (data-ink ratio), подписи значений вместо легенд.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data import CRIT_ORDER, FP

GREY = "#9AA0A6"
DARK = "#3C4043"
ACCENT = "#1A73E8"
SIGNAL = "#D93025"
FONT = dict(family="Inter, Arial, sans-serif", size=13, color=DARK)


def _layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        font=FONT,
        height=height,
        margin=dict(l=10, r=20, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#DADCE0")
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def _hbar_room(fig: go.Figure, values) -> go.Figure:
    """Скрывает ось X и оставляет место справа под подписи значений."""
    top = max(values, default=0) or 1
    fig.update_xaxes(visible=False, range=[0, top * 1.3])
    return fig


def daily_dynamics(df: pd.DataFrame) -> go.Figure:
    """Приток карточек по дням: сырые значения фоном, скользящее среднее за 7 дней — сигналом."""
    daily = df.groupby("Дата").size().asfreq("D", fill_value=0) if len(df) else pd.Series(dtype=int)
    fig = go.Figure()
    fig.add_bar(x=daily.index, y=daily.values, marker_color="#DADCE0", name="за день",
                hovertemplate="%{x|%d.%m.%Y}: %{y} карточек<extra></extra>")
    fig.add_scatter(x=daily.index, y=daily.rolling(7, min_periods=1).mean(), mode="lines",
                    line=dict(color=ACCENT, width=2), name="среднее за 7 дн.",
                    hovertemplate="%{x|%d.%m.%Y}: %{y:.0f} в среднем<extra></extra>")
    _layout(fig, "Приток карточек в день (линия — среднее за 7 дней)")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F3F4")
    fig.update_layout(bargap=0)
    return fig


def verdict_bars(df: pd.DataFrame) -> go.Figure:
    """Распределение вердиктов аналитиков: доля ложных срабатываний выделена."""
    counts = df["Вердикт"].value_counts().sort_values()
    colors = [SIGNAL if v == FP else GREY for v in counts.index]
    share = counts / counts.sum() if len(counts) else counts
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h", marker_color=colors,
        text=[f"{c:,} ({s:.0%})".replace(",", " ") for c, s in zip(counts.values, share.values)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fp_share = share.get(FP, 0.0)
    _layout(fig, f"Вердикты аналитиков: ложных срабатываний {fp_share:.0%}".replace("%", " %"))
    _hbar_room(fig, counts.values)
    # Не даем 1–2 категориям растянуться на всю высоту панели.
    fig.update_yaxes(range=[-0.5, max(len(counts), 5) - 0.5])
    return fig


def criticality_bars(df: pd.DataFrame) -> go.Figure:
    """Карточки по критичности в порядке шкалы; красный — только у «Критической»."""
    counts = df["Критичность"].value_counts().reindex(CRIT_ORDER, fill_value=0)[::-1]
    colors = [SIGNAL if c == "Критическая" else GREY for c in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h", marker_color=colors,
        text=[f"{c:,}".replace(",", " ") for c in counts.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    _layout(fig, "Карточки по критичности")
    _hbar_room(fig, counts.values)
    return fig


def category_bars(df: pd.DataFrame, top: int = 8) -> go.Figure:
    """Топ категорий IRP: столбец — число карточек, подпись — доля ложных срабатываний."""
    g = df.groupby("Пиктограмма IRP")["Вердикт"].agg(count="size", fp=lambda s: (s == FP).mean())
    g = g.nlargest(top, "count").sort_values("count")
    fig = go.Figure(go.Bar(
        x=g["count"], y=g.index, orientation="h", marker_color=GREY,
        text=[f"{c:,}  · FP {f:.0%}".replace(",", " ") for c, f in zip(g["count"], g["fp"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    _layout(fig, f"Топ-{top} категорий IRP (подпись — доля ложных срабатываний)")
    _hbar_room(fig, g["count"].values)
    return fig


def hourly_profile(df: pd.DataFrame) -> go.Figure:
    """Суточный профиль создания карточек (UTC)."""
    hours = df["Час"].value_counts().reindex(range(24), fill_value=0)
    peak = hours.idxmax() if hours.sum() else -1
    colors = [ACCENT if h == peak else GREY for h in hours.index]
    fig = go.Figure(go.Bar(x=hours.index, y=hours.values, marker_color=colors,
                           hovertemplate="%{x}:00 — %{y} карточек<extra></extra>"))
    _layout(fig, "Суточный профиль, UTC (выделен пиковый час)")
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)), ticksuffix=":00")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F3F4")
    return fig


# ---- Модель-советчик (3.6)

def confusion_matrix_fig(tp: int, fn: int, fp: int, tn: int) -> go.Figure:
    """Матрица ошибок: строки — факт, столбцы — предсказание (как в лекции 3.1).
    Цвет — доля строки: так видно, какую часть каждого факта модель отнесла верно."""
    rows = [[tp, fn], [fp, tn]]
    share = [[v / max(sum(r), 1) for v in r] for r in rows]
    labels = [["TP · найдено", "FN · пропущено"], ["FP · ложная тревога", "TN · корректное молчание"]]
    text = [[f"{labels[i][j]}<br><b>{rows[i][j]:,}</b> ({share[i][j]:.0%})".replace(",", " ")
             for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=share, x=["Предсказано: инцидент", "Предсказано: FP"], y=["Факт: инцидент", "Факт: FP"],
        text=text, texttemplate="%{text}", colorscale=[[0, "#F8F9FA"], [1, "#BDC1C6"]], zmin=0, zmax=1,
        showscale=False, hoverinfo="skip", xgap=3, ygap=3,
    ))
    _layout(fig, "Матрица ошибок на отложенной выборке", height=300)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top", linecolor="white")
    fig.update_layout(margin=dict(l=10, r=10, t=80, b=10))
    return fig


def pr_curve_fig(precision, recall, point_p: float, point_r: float) -> go.Figure:
    """Кривая точности и полноты с рабочей точкой текущего порога."""
    fig = go.Figure()
    fig.add_scatter(x=recall, y=precision, mode="lines", line=dict(color=GREY, width=2),
                    hovertemplate="полнота %{x:.0%} · точность %{y:.0%}<extra></extra>")
    fig.add_scatter(x=[point_r], y=[point_p], mode="markers+text", marker=dict(color=ACCENT, size=11),
                    text=[f"P {point_p:.0%} · R {point_r:.0%}  "], textposition="top left",
                    hoverinfo="skip")
    _layout(fig, "Кривая точности и полноты (точка — текущий порог)", height=300)
    fig.update_xaxes(title="Полнота", tickformat=".0%", range=[0, 1.02])
    fig.update_yaxes(title="Точность", tickformat=".0%", range=[0, 1.02], showgrid=True, gridcolor="#F1F3F4")
    return fig


def score_histogram(scores, threshold: float) -> go.Figure:
    """Распределение модельных оценок в очереди; справа от порога — карточки «к разбору»."""
    fig = go.Figure()
    above = scores[scores >= threshold]
    below = scores[scores < threshold]
    bins = dict(start=0, end=1, size=0.02)
    fig.add_histogram(x=below, xbins=bins, marker_color="#DADCE0", hovertemplate="%{y}<extra></extra>")
    fig.add_histogram(x=above, xbins=bins, marker_color=SIGNAL, hovertemplate="%{y}<extra></extra>")
    fig.add_vline(x=threshold, line=dict(color=DARK, width=1, dash="dot"),
                  annotation_text=f"порог {threshold:.2f}", annotation_position="top")
    _layout(fig, "Оценки модели по очереди без вердикта (красным — к разбору; шкала логарифмическая)", height=300)
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_xaxes(title="Модельная оценка «инцидент»", range=[0, 1])
    fig.update_yaxes(type="log", title="Карточек", showgrid=True, gridcolor="#F1F3F4")
    return fig


# ---- Объяснения SHAP (3.7)

def _short(text: str, n: int = 38) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _proba(logit: float) -> float:
    return 1 / (1 + np.exp(-logit))


def shap_waterfall(contrib: pd.DataFrame, base: float, top: int = 10, title: str = "") -> go.Figure:
    """Waterfall локального объяснения: от средней оценки модели к оценке этой карточки.
    Красный — вклад в сторону «инцидент», синий — в сторону «ложное срабатывание»."""
    head, rest = contrib.iloc[:top], contrib.iloc[top:]
    labels = [f"{_short(f'{f} = {v}')}" for f, v in zip(head["Признак"], head["Значение"])]
    values = head["Вклад"].tolist()
    if len(rest):
        labels.append(f"остальные {len(rest)} признаков")
        values.append(float(rest["Вклад"].sum()))
    # Снизу вверх: база → вклады от меньших к большим → итог; читается сверху от самого сильного.
    labels, values = labels[::-1], values[::-1]
    fx = base + sum(values)
    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=["средняя оценка E[f(x)]"] + labels + ["оценка карточки f(x)"],
        x=[base] + values + [0],
        measure=["absolute"] + ["relative"] * len(values) + ["total"],
        base=0,
        increasing=dict(marker=dict(color=SIGNAL)),
        decreasing=dict(marker=dict(color=ACCENT)),
        totals=dict(marker=dict(color=DARK)),
        text=[f"{base:+.2f}"] + [f"{v:+.2f}" for v in values] + [f"{fx:+.2f}"],
        textposition="outside",
        connector=dict(line=dict(color="#DADCE0", width=1)),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    _layout(fig, title or f"Почему такая оценка: {_proba(base):.0%} в среднем → {_proba(fx):.0%} у карточки",
            height=110 + 30 * (len(values) + 2))
    fig.update_xaxes(title="Лог-шансы класса «инцидент»", showgrid=True, gridcolor="#F1F3F4", zeroline=True,
                     zerolinecolor="#DADCE0")
    fig.update_layout(margin=dict(l=10, r=40, t=50, b=10))
    return fig


def shap_importance(values: np.ndarray, features, top: int = 15) -> go.Figure:
    """Глобальная важность: средний модуль SHAP-вклада по выборке карточек."""
    imp = pd.Series(np.abs(values).mean(axis=0), index=features).nlargest(top).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h", marker_color=GREY,
        text=[f"{v:.2f}" for v in imp.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    _layout(fig, f"Важность признаков: средний |SHAP| по {len(values):,} карточкам".replace(",", " "),
            height=120 + 26 * len(imp))
    _hbar_room(fig, imp.values)
    return fig


def shap_by_value(values: np.ndarray, data: np.ndarray, feature: str, features, top: int = 12,
                  min_count: int = 20) -> go.Figure:
    """Вклад значений одного признака: средний SHAP по каждому частому значению (с числом карточек).
    Значения реже min_count не показываем: среднее по 1–2 карточкам ничего не говорит о модели."""
    j = list(features).index(feature)
    df = pd.DataFrame({"value": data[:, j], "shap": values[:, j]})
    g = df.groupby("value")["shap"].agg(["mean", "size"])
    g = g[g["size"] >= min_count].nlargest(top, "size").sort_values("mean")
    colors = [SIGNAL if v > 0 else ACCENT for v in g["mean"]]
    fig = go.Figure(go.Bar(
        x=g["mean"], y=[_short(str(v), 34) for v in g.index], orientation="h", marker_color=colors,
        text=[f"{m:+.2f} · n={n}" for m, n in zip(g["mean"], g["size"])], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    _layout(fig, f"Вклад значений признака «{feature}» (средний SHAP; значения, встречающиеся ≥ {min_count} раз)",
            height=120 + 28 * len(g))
    lim = max(float(g["mean"].abs().max()) if len(g) else 0.0, 0.1) * 1.6
    fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor="#9AA0A6", showticklabels=False)
    return fig
