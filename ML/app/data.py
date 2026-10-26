"""Загрузка и подготовка карточек SOC из parquet-датасета курса."""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "lecture" / "Dataset" / "iavd_incidents_raw_target.parquet"
)

# Из 97 полей берем только те, что нужны интерфейсу: так датасет занимает ~0,3 ГБ вместо ~1 ГБ.
COLUMNS = [
    "EntityId",
    "Создан",
    "Вердикт",
    "verdict",
    "Критичность",
    "Группа устранения",
    "Пиктограмма IRP",
    "Алерт",
    "Hostname Цели",
    "processName",
    "Ответственный",
    "Время решения",
]

CRIT_ORDER = ["Критическая", "Высокая", "Средняя", "Низкая", "Отсутствует"]
NO_VERDICT = "Не указан"
FP = "False Positive"


@lru_cache(maxsize=1)
def load_cards() -> pd.DataFrame:
    """Читает датасет один раз на процесс; путь можно переопределить переменной IAVD_DATA."""
    path = Path(os.environ.get("IAVD_DATA", DEFAULT_PATH))
    df = pd.read_parquet(path, columns=COLUMNS)
    # Время хранится в UTC; контрольные числа банка EDA (пик в 8:00) посчитаны в UTC.
    df["Создан"] = df["Создан"].dt.tz_convert(None)
    df["Дата"] = df["Создан"].dt.normalize()
    df["Час"] = df["Создан"].dt.hour
    df["Время решения, ч"] = df["Время решения"] / 3600
    df["Критичность"] = pd.Categorical(df["Критичность"], categories=CRIT_ORDER, ordered=True)
    return df.sort_values("Создан", ascending=False, ignore_index=True)


def kpis(df: pd.DataFrame) -> dict:
    """Ключевые показатели для верхней строки дашборда."""
    total = len(df)
    return {
        "total": total,
        "fp_share": (df["Вердикт"] == FP).mean() if total else 0.0,
        "no_verdict": int((df["Вердикт"] == NO_VERDICT).sum()),
        "critical_open": int(((df["Критичность"] == "Критическая") & (df["Вердикт"] == NO_VERDICT)).sum()),
        "median_hours": df["Время решения, ч"].median() if total else 0.0,
    }


TABLE_COLUMNS = ["EntityId", "Создан", "Критичность", "Вердикт", "Пиктограмма IRP", "Группа устранения", "Алерт"]


def table_view(df: pd.DataFrame) -> pd.DataFrame:
    """Компактное представление для таблицы: без служебных полей."""
    view = df[TABLE_COLUMNS].copy()
    view["Критичность"] = view["Критичность"].astype(str)
    return view
