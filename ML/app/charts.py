"""Графики Plotly для дашборда SOC.

Правила оформления (лекции 2.1–2.3): одна нейтральная серая палитра, насыщенный цвет
несет один смысл; красный — только «требует реакции» (критичность, ложные тревоги как нагрузка);
сетка и рамки сведены к минимуму (data-ink ratio), подписи значений вместо легенд.
"""
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
