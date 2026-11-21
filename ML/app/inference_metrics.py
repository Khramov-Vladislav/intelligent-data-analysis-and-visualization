"""Матрицы ошибок для всех порогов и разрезы пропущенных инцидентов."""
import numpy as np
import pandas as pd


def threshold_counts(scores: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Оценка ≥ порога. Карточки без бинарной метки учитываются только в нагрузке."""
    flags = pd.DataFrame({"TP": labels.eq(1), "FP": labels.eq(0), "flagged": 1}, index=scores.index)
    counts = flags.groupby(scores).sum().sort_index(ascending=False).cumsum()
    if not counts.empty:
        # Точка перед самой высокой оценкой соответствует нулю карточек к разбору.
        counts.loc[max(1.0, np.nextafter(counts.index[0], np.inf))] = 0
        counts = counts.sort_index(ascending=False)
    counts.index.name = "threshold"
    counts["FN"] = int(labels.eq(1).sum()) - counts["TP"]
    counts["TN"] = int(labels.eq(0).sum()) - counts["FP"]
    return counts


def incidents_by(labels: pd.Series, missed: pd.Series, key: pd.Series) -> pd.DataFrame:
    """При отсутствии реальных инцидентов доля пропусков не определяется."""
    incidents = labels.eq(1)
    counts = pd.DataFrame({"incidents": incidents, "missed": missed & incidents}).groupby(key).sum()
    counts["rate"] = counts["missed"] / counts["incidents"].where(counts["incidents"].gt(0))
    return counts
