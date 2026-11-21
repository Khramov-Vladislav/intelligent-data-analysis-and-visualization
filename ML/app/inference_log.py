"""Подготовка учебного лога для анализа качества сохранённых прогнозов."""
import os
from pathlib import Path

import pandas as pd

from course_model import MATERIALS

LOG_THRESHOLD = 0.18
SRS_THRESHOLD = 0.00498
LOG_PATH = (
    MATERIALS / "Datasets" / "Задание 2: Инференс модели" / "Готовые данные"
    / "MODEL_PREDICTIONS_target.parquet"
)
SEGMENTS = {
    "Вендор SIEM": "Вендор из сием",
    "Конвейер SIEM": "Конвейер SIEM",
    "Категория IRP": "Пиктограмма IRP",
    "Критичность": "Критичность",
}


def log_path() -> Path:
    return Path(os.environ.get("IAVD_INFERENCE_LOG", LOG_PATH)).expanduser()


def load_log(path: Path) -> pd.DataFrame:
    log = pd.read_parquet(path, columns=[
        "Дата", "version", "Flag_ValidLabel", "y", "y_pred_proba", *SEGMENTS.values(),
    ])
    if log.empty or log["version"].isna().any() or log["version"].nunique() != 1:
        raise ValueError("Нужен непустой лог одной версии модели.")
    log = log.rename(columns={"y_pred_proba": "score"})
    if not log["score"].between(0, 1).all():
        raise ValueError("В логе есть оценки вне диапазона от 0 до 1 или пустые оценки.")

    valid = log.pop("Flag_ValidLabel").eq(1)
    if not log.loc[valid, "y"].isin([0, 1]).all():
        raise ValueError("У размеченных карточек метка y должна быть равна 0 или 1.")
    log["y"] = log["y"].where(valid)
    log["Дата"] = pd.to_datetime(log["Дата"]).dt.normalize()
    if log["Дата"].isna().any():
        raise ValueError("В логе есть прогнозы без даты.")
    log["Неделя"] = log["Дата"] - pd.to_timedelta(log["Дата"].dt.dayofweek, unit="D")
    for field in SEGMENTS.values():
        log[field] = log[field].astype("string").str.strip().replace("", pd.NA).fillna("Не указан")
    return log
