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
SHAP_SAMPLE = MODEL_DIR / "advisor_baseline_shap.npz"  # предрасчитанные SHAP-значения для глобальных графиков

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


# ---- Объяснения (SHAP)

_explainers: dict[int, object] = {}


def _split(model):
    """Пайплайн «кодирование → бустинг» объясняем по бустингу на закодированных признаках;
    модель без шага кодирования — как есть."""
    steps = getattr(model, "named_steps", {})
    if "encode" in steps and "clf" in steps:
        return steps["encode"].transform, steps["clf"]
    return (lambda X: X), model


def explain_batch(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP-значения в лог-шансах класса «инцидент» и базовое значение (средний выход модели).
    Сумма базы и вкладов строки равна decision_function модели для этой карточки."""
    import shap

    transform, est = _split(model)
    explainer = _explainers.get(id(est))
    if explainer is None:
        explainer = _explainers[id(est)] = shap.TreeExplainer(est)
    values = explainer.shap_values(transform(X))
    if isinstance(values, list):  # старые версии shap: список по классам
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (карточки, признаки, классы)
        values = values[:, :, 1]
    base = np.ravel(explainer.expected_value)
    return values, float(base[-1])


def explain_card(model, X_row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Локальное объяснение одной карточки: признак, его значение и вклад, по убыванию |вклада|."""
    values, base = explain_batch(model, X_row)
    out = pd.DataFrame({
        "Признак": X_row.columns,
        "Значение": X_row.iloc[0].astype(str).replace("nan", "—").to_numpy(),
        "Вклад": values[0],
    })
    return out.reindex(out["Вклад"].abs().sort_values(ascending=False).index).reset_index(drop=True), base


# ---- Частичная зависимость (PDP / ICE)

def pdp_grid(X: pd.DataFrame, feature: str, points: int = 20, top: int = 12) -> list:
    """Сетка значений признака: для числовых — квантили 5–95 % (без экстраполяции за пределы
    основной массы данных), для категориальных — самые частые значения."""
    col = X[feature]
    if feature in NUMERIC:
        qs = col.dropna().quantile(np.linspace(0.05, 0.95, points)).round(2)
        return sorted(qs.unique().tolist())
    return col.value_counts().head(top).index.tolist()


def partial_dependence(model, X: pd.DataFrame, feature: str, grid: list) -> tuple[np.ndarray, np.ndarray]:
    """PDP и ICE: каждой карточке выборки по очереди подставляем значение из сетки и усредняем оценку.
    Возвращает (pdp[len(grid)], ice[n_cards, len(grid)]) в шкале модельной оценки «инцидент»."""
    ice = np.empty((len(X), len(grid)))
    for k, value in enumerate(grid):
        X_mod = X.copy()
        X_mod[feature] = value
        ice[:, k] = predict_scores(model, X_mod)
    return ice.mean(axis=0), ice
