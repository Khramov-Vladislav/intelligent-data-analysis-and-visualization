"""Карточки, готовые вклады SHAP и контрольные расчёты практики 3.7."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import expit

from course_model import COURSE_VERSION

TOLERANCE = 1e-6
TEAM_COLUMNS = ["EntityId", "Вердикт", "Оценка", "Исход"]
CONTRIB_COLUMNS = ["Признак", "Источник", "Значение", "Вклад"]
TEST_IMPORTANCE = "IAVD_3.7_BI_team/data/5/Parquet/Gen/fact_importance.parquet"


def read_cards_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig").set_index("EntityId", verify_integrity=True)


def test_importance(kit: Path) -> pd.Series:
    """Готовая важность на тесте из того же прогона модели."""
    archive = kit.parent / "IAVD_3.7_BI_team.zip"
    if not archive.is_file():
        return pd.Series(dtype=float)
    with ZipFile(archive) as bundle:
        table = pd.read_parquet(BytesIO(bundle.read(TEST_IMPORTANCE)))
    test = table.loc[table["sample"].eq("test") & table["model"].eq(COURSE_VERSION)]
    return test.set_index("source_column")["mean_abs_shap"]


class CourseShap:
    def __init__(self, kit, model):
        from course_ml.data import FREQ_FIELDS, prepare
        from course_ml.train import feature_frame

        pack = kit / "course_pack"
        self.cards = read_cards_csv(pack / "cards_practice.csv")
        shap = read_cards_csv(pack / "shap_cards.csv").loc[self.cards.index]
        raw = pd.read_parquet(pack / "cards_raw.parquet")
        raw["EntityId"] = raw["EntityId"].astype(int)
        self.raw = raw.set_index("EntityId", drop=False, verify_integrity=True).loc[self.cards.index]

        self.threshold = float(model.threshold)
        self.base = shap["base"]
        self.values = shap.drop(columns=["y_true", "proba", "outcome", "base"])
        self.log_odds = self.base + self.values.sum(axis=1)
        self.probability = expit(self.log_odds)
        self.csv_error = (self.probability - self.cards["proba"]).abs()

        self.prepared, _, _ = prepare(self.raw, model.cfg)
        for column, source in FREQ_FIELDS.items():
            self.prepared[column] = self.prepared[source].astype(str).map(model.freq_maps[source]).fillna(0.0)
        matrix = model.preproc.transform(feature_frame(self.prepared, model.cfg))
        raw_score = model.model.predict(np.asarray(matrix, dtype=np.float32), raw_score=True)
        self.raw_error = (self.log_odds - raw_score).abs()
        positive = self.cards["y_true"].eq(1)
        predicted = self.probability.ge(self.threshold)
        outcomes = np.where(positive, np.where(predicted, "TP", "FN"), np.where(predicted, "FP", "TN"))
        if (
            self.csv_error.max() > TOLERANCE
            or self.raw_error.max() > TOLERANCE
            or not self.cards["outcome"].eq(outcomes).all()
        ):
            raise ValueError("Вклады SHAP или исходы не совпали с контрольными данными курса.")

        self.practice_importance = self.values.abs().mean()
        self.test_importance = test_importance(kit)

    def team(self, number: int) -> pd.DataFrame:
        cards = self.cards.loc[self.cards["team"].eq(number)].sort_values("rank_in_team")
        return pd.DataFrame({
            "EntityId": cards.index,
            "Вердикт": self.raw.loc[cards.index, "Вердикт"].to_numpy(),
            "Оценка": cards["proba"].to_numpy(),
            "Исход": cards["outcome"].to_numpy(),
        })

    def contributions(self, entity_id: int) -> pd.DataFrame:
        values = self.values.loc[entity_id]
        values = values.reindex(values.abs().sort_values(ascending=False, kind="stable").index)
        raw = self.raw.loc[entity_id]
        prepared = self.prepared.loc[entity_id]
        return pd.DataFrame({
            "Признак": values.index,
            "Источник": ["Поле карточки" if name in raw.index else "Сконструирован" for name in values.index],
            "Значение": [display_value(prepared.get(name, raw.get(name))) for name in values.index],
            "Вклад": values.to_numpy(),
        })


def display_value(value, limit=120) -> str:
    if value is None or pd.isna(value):
        return "–"
    result = " ".join(str(value).split())
    return (result if len(result) <= limit else result[:limit - 1] + "…") or "–"
