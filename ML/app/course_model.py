"""Загрузка модели курса и её исходных полей для практики 3.6."""

import os
import sys
from functools import lru_cache
from pathlib import Path

MATERIALS = Path(__file__).resolve().parents[2] / "materials"
COURSE_DATA = (
    MATERIALS / "Datasets" / "Задание 2: Инференс модели"
    / "Под обучение" / "iavd_ml_dataset.parquet"
)
COURSE_VERSION = "lgbm_gate1__v001"


@lru_cache(maxsize=1)
def configured_course():
    """Кэширует загрузку и ошибку, чтобы интерфейс мог показать её при запуске."""
    value = os.environ.get("IAVD_COURSE_KIT")
    if not value:
        return None, None, None
    kit = Path(value).expanduser().resolve()
    try:
        if not (kit / "course_ml" / "inference.py").is_file():
            raise FileNotFoundError(f"В {kit} не найден пакет course_ml.")
        sys.path.insert(0, str(kit))
        from course_ml.inference import CourseModel

        return kit, CourseModel(kit / "models" / COURSE_VERSION), None
    except Exception as error:
        return kit, None, error


def load_course_cards(model):
    """Читает признаки модели и поля, нужные для очереди и оценки качества."""
    import pandas as pd
    from course_ml.data import LIST_LEN_SOURCE

    fields = model.fields
    columns = {field for kind in ("categorical", "text", "ip") for field in fields.get(kind, [])}
    times = model.cfg["dataset"]["time"]
    columns.update(times[name] for name in ("column", "first_event", "last_event"))
    columns.update(LIST_LEN_SOURCE)
    columns.update((
        "EntityId", "Вердикт", "Критичность", "Пиктограмма IRP", "Наименование",
        "Правило корреляции SIEM", "Список вредоносных команд",
    ))
    path = Path(os.environ.get("IAVD_COURSE_DATA", COURSE_DATA)).expanduser()
    cards = pd.read_parquet(path, columns=sorted(columns))
    model.check(cards)
    return cards
