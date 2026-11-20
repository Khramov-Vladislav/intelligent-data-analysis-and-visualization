"""Загрузка и подготовка карточек SOC из parquet-датасета курса."""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pyarrow.parquet import read_schema

DEFAULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "materials"
    / "Datasets"
    / "Задание 1: Датасет карточек сок + календарь"
    / "iavd_incidents_raw_target.parquet"
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
RULE = "Правило корреляции SIEM"
TAKEN_AT = "Дата взятия в работу"
TAKE_SLA = "SLA по взятию в работу Атаки"
TAKE_OVERDUE = "SLA просрочен (взятие в работу)"
EDA_COLUMNS = [RULE, TAKEN_AT, TAKE_SLA, TAKE_OVERDUE]


def dataset_path() -> Path:
    """Путь к датасету; переопределяется переменной IAVD_DATA."""
    return Path(os.environ.get("IAVD_DATA", DEFAULT_PATH))


@lru_cache(maxsize=1)
def load_cards() -> pd.DataFrame:
    """Читает датасет один раз на процесс."""
    path = dataset_path()
    available = set(read_schema(path).names)
    columns = COLUMNS + [name for name in EDA_COLUMNS if name in available]
    df = pd.read_parquet(path, columns=columns).reindex(columns=COLUMNS + EDA_COLUMNS)
    # Время хранится в UTC; контрольные числа банка EDA (пик в 8:00) посчитаны в UTC.
    df["Создан"] = df["Создан"].dt.tz_convert(None)
    df["Дата"] = df["Создан"].dt.normalize()
    df["Час"] = df["Создан"].dt.hour
    # Нагрузка часа относится ко всему потоку, даже когда аналитик смотрит отдельную группу.
    hour = df["Создан"].dt.floor("h")
    df["Карточек в часе"] = hour.groupby(hour).transform("size")
    taken_at = pd.to_datetime(df[TAKEN_AT], utc=True).dt.tz_convert(None)
    take_minutes = (taken_at - df["Создан"]).dt.total_seconds() / 60
    df["До взятия, мин"] = take_minutes.where(take_minutes.ge(0))
    sla_minutes = pd.to_numeric(df[TAKE_SLA], errors="coerce") / 60_000
    df["SLA взятия, мин"] = sla_minutes.where(sla_minutes.gt(0))
    df["Флаг просрочки взятия"] = df[TAKE_OVERDUE].astype("string").str.lower().map({"true": 1, "false": 0})
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


def filter_cards(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    crits: list[str],
    groups: list[str],
    categories: list[str],
    only_open: bool,
) -> pd.DataFrame:
    """Срез карточек по фильтрам; пустой список в фильтре означает «все значения»."""
    mask = df["Дата"].between(start, end)
    if crits:
        mask &= df["Критичность"].isin(crits)
    if groups:
        mask &= df["Группа устранения"].isin(groups)
    if categories:
        mask &= df["Пиктограмма IRP"].isin(categories)
    if only_open:
        mask &= df["Вердикт"] == NO_VERDICT
    return df[mask]


def options(df: pd.DataFrame, column: str) -> list[str]:
    """Значения для селектора, от частых к редким."""
    return df[column].dropna().value_counts().index.astype(str).tolist()


TABLE_COLUMNS =["EntityId", "Создан", "Критичность", "Вердикт", "Пиктограмма IRP", "Группа устранения", "Алерт"]


def table_view(df: pd.DataFrame) -> pd.DataFrame:
    """Компактное представление для таблицы: без служебных полей."""
    view = df[TABLE_COLUMNS].reset_index(drop=True)
    view["Критичность"] = view["Критичность"].astype(str)
    return view
