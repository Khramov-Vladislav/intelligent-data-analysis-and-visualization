"""Модель-советчик: признаки, разметка, загрузка .joblib и пакетный инференс.

Модуль общий для обучения (model/train_baseline.py) и приложения: признаки должны
строиться одинаково, иначе модель получит на инференсе не те данные, что при обучении.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "~~||~~"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DEFAULT_MODEL = MODEL_DIR / "advisor_baseline.joblib"
TEST_IDS = MODEL_DIR / "advisor_baseline_test_ids.npy"  # EntityId отложенной выборки для оценки в приложении

# Класс 1 — реальный инцидент, класс 0 — ложное срабатывание (обоснование — СРС к 3.2).
POSITIVE_VERDICTS = ["Решён", "Не решён", "Закрыт по совокупности инцидентов", "Закрыт из родительской атаки"]
NEGATIVE_VERDICT = "False Positive"

# Только поля, известные в момент создания карточки. Группа устранения, флаги процесса
# (Спам, Проверка на FP, Ложное срабатывание), «Решение» и времена обработки заполняются
# по ходу разбора и дали бы утечку таргета.
CATEGORICAL = [
    "Критичность",
    "Критичность инцидента в источнике",
    "Тип инцидента",
    "Источник инцидента",
    "Вендор из сием",
    "Конвейер SIEM",
    "Наименование",
    "Пиктограмма IRP",
    "categoryHigh",
    "categoryLow",
    "categoryMain",
    "processName",
    "processParentName",
    "object",
    "Привилегии аккаунта",
    "Теги",
]
NUMERIC = [
    "Час",
    "День недели",
    "Количество инцидентов",
    "Алертов в карточке",
    "Значений инцидента",
    "Длина команды",
    "Есть powershell",
]
FEATURES = CATEGORICAL + NUMERIC
RAW_COLUMNS = sorted(
    set(CATEGORICAL) - {"Час", "День недели"}
    | {"EntityId", "Создан", "Вердикт", "Количество инцидентов", "Алерт", "Значение Инцидента (группировка)", "Команда"}
)


def _first(s: pd.Series) -> pd.Series:
    """Первый элемент мультизначного поля: составные значения карточки дают редкие уникальные строки."""
    return s.astype("string").str.split(SEP, n=1, regex=False).str[0]


def _count(s: pd.Series) -> pd.Series:
    return s.astype("string").str.count(SEP.replace("|", r"\|")).add(1).fillna(0)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Таблица признаков X из сырых полей карточки."""
    X = pd.DataFrame(index=raw.index)
    for col in CATEGORICAL:
        X[col] = _first(raw[col]).fillna("—").astype(str)
    created = pd.to_datetime(raw["Создан"])
    X["Час"] = created.dt.hour
    X["День недели"] = created.dt.dayofweek
    X["Количество инцидентов"] = pd.to_numeric(raw["Количество инцидентов"], errors="coerce")
    X["Алертов в карточке"] = _count(raw["Алерт"])
    X["Значений инцидента"] = _count(raw["Значение Инцидента (группировка)"])
    cmd = raw["Команда"].astype("string")
    X["Длина команды"] = cmd.str.len().fillna(0)
    X["Есть powershell"] = cmd.str.contains("powershell", case=False, regex=False).fillna(False).astype(int)
    return X[FEATURES]


@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """Сырые поля, из которых строятся признаки; читаются один раз на процесс."""
    from data import dataset_path

    return pd.read_parquet(dataset_path(), columns=RAW_COLUMNS)


def labels(raw: pd.DataFrame) -> pd.Series:
    """y: 1 — инцидент, 0 — FP, NaN — нет эталона («Не указан», «Дубликат»)."""
    y = pd.Series(np.nan, index=raw.index)
    y[raw["Вердикт"].isin(POSITIVE_VERDICTS)] = 1
    y[raw["Вердикт"] == NEGATIVE_VERDICT] = 0
    return y


def model_path() -> Path:
    """Путь к .joblib: модель курса подключается переменной IAVD_MODEL без правки кода."""
    return Path(os.environ.get("IAVD_MODEL", DEFAULT_MODEL))


def load_model():
    """Загружает модель и сопроводительные метаданные (порог, отсечка обучения, метрики)."""
    import joblib

    path = model_path()
    model = joblib.load(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.setdefault("threshold", 0.5)
    meta.setdefault("name", path.name)
    return model, meta


def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Модельная оценка класса «инцидент» (y_pred_proba)."""
    return model.predict_proba(X)[:, 1]
