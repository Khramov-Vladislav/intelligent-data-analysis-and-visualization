"""Карточки эксплуатации и предрассчитанные вклады модели из комплекта 3.7."""
from zipfile import ZipFile

import pandas as pd
from scipy.special import expit, logit

from course_model import COURSE_VERSION
from course_shap import TOLERANCE, read_bi_table
from inference_metrics import incidents_by

GROUPING = "Значение Инцидента (группировка)"
GROUPING_FEATURE = f"len_{GROUPING}"
SOURCE_FEATURE = "src_is_private"
CATEGORY = "Категория IRP"
OUTCOMES = {"FN": "Пропущен", "TP": "Найден", "FP": "Ложная тревога", "TN": "Верное молчание"}
FEATURES_TO_COMPARE = {
    "Адрес источника": (SOURCE_FEATURE, ["Внешний", "Внутренний", "Смешанный", "Не указан"]),
    "Поле группировки": (GROUPING_FEATURE, ["Пустое", "Заполнено"]),
}


class ShapFlow:
    def __init__(self, archive_path, raw_path):
        from course_ml.data import ips, is_private, json_to_text, parse_json_list

        with ZipFile(archive_path) as bundle:
            metadata = read_bi_table(bundle, "dim_models", ["model", "tau"]).set_index("model")
            self.threshold = float(metadata.loc[COURSE_VERSION, "tau"])
            self.logit_threshold = float(logit(self.threshold))
            fields = read_bi_table(bundle, "dim_features", ["source_column"]).set_index("source_column")
            shap = read_bi_table(bundle, "fact_shap", ["entity_id", "model", "split", "source_column", "shap", "base"])
            scores = read_bi_table(bundle, "fact_score", ["entity_id", "model", "split", "score", "y_pred"])
            labels = read_bi_table(bundle, "dim_cards", ["EntityId", "y", "verdict", "rule", "criticality"])

        shap = shap.loc[shap["model"].eq(COURSE_VERSION) & shap["split"].eq("productive_20584")]
        scores = scores.loc[scores["model"].eq(COURSE_VERSION) & scores["split"].eq("productive")]
        scores = scores.set_index("entity_id", verify_integrity=True)
        self.values = shap.pivot(index="entity_id", columns="source_column", values="shap")
        if (
            self.values.empty
            or set(scores.index) != set(self.values.index)
            or set(self.values.columns) != set(fields.index)
            or self.values.isna().any().any()
            or not shap.groupby("entity_id")["base"].nunique(dropna=False).eq(1).all()
        ):
            raise ValueError("Оценки, база или набор вкладов различаются между таблицами курса.")

        labels = labels.set_index("EntityId", verify_integrity=True).loc[self.values.index]
        cards = scores.reindex(self.values.index).join(labels)
        cards["rule"] = cards["rule"].map(json_to_text).fillna("Не указано")
        cards["base"] = shap.groupby("entity_id")["base"].first()
        cards["z"] = cards["base"] + self.values.sum(axis=1)
        cards["error"] = (expit(cards["z"]) - cards["score"]).abs()
        predicted = cards["score"].ge(self.threshold)
        if (
            not cards["error"].le(TOLERANCE).all()
            or not predicted.eq(cards["y_pred"]).all()
            or not predicted.eq(cards["z"].ge(self.logit_threshold)).all()
        ):
            raise ValueError("Вклады SHAP не воспроизводят оценку или решение модели.")
        cards["outcome"] = pd.Series(pd.NA, index=cards.index, dtype="string")
        for label, decision, outcome in ((1, True, "TP"), (1, False, "FN"), (0, True, "FP"), (0, False, "TN")):
            cards.loc[cards["y"].eq(label) & predicted.eq(decision), "outcome"] = outcome

        raw = pd.read_parquet(raw_path, columns=["EntityId", "Пиктограмма IRP", "src", GROUPING])
        raw["EntityId"] = raw["EntityId"].astype(str)
        raw = raw.set_index("EntityId", verify_integrity=True).loc[cards.index.astype(str)]
        raw.index = cards.index
        cards[CATEGORY] = raw["Пиктограмма IRP"].map(json_to_text).fillna("Не указана")
        cards["Адрес источника"] = raw["src"].map(
            lambda value: source_kind(ips(json_to_text(value)), is_private)
        )
        cards["Элементов группировки"] = raw[GROUPING].map(lambda value: len(parse_json_list(value)))
        cards["Поле группировки"] = cards["Элементов группировки"].eq(0).map({True: "Пустое", False: "Заполнено"})
        self.cards = cards

    def evidence(self, cards: pd.DataFrame, name: str) -> pd.DataFrame:
        feature, order = FEATURES_TO_COMPARE[name]
        labelled = cards.loc[cards["outcome"].notna()]
        evidence = labelled.assign(contribution=self.values.loc[labelled.index, feature]).groupby(name).agg(
            count=("y", "size"), incident_rate=("y", "mean"), contribution=("contribution", "mean"),
        )
        return evidence.reindex([value for value in order if value in evidence.index])

    def misses_by_category(self) -> pd.DataFrame:
        cards = self.cards
        groups = incidents_by(cards["y"], cards["outcome"].eq("FN").fillna(False), cards[CATEGORY])
        return groups.loc[groups["incidents"].gt(0)]

    def contributions(self, card_id) -> pd.DataFrame:
        values = self.values.loc[card_id]
        values = values.reindex(values.abs().sort_values(ascending=False, kind="stable").index)
        card = self.cards.loc[card_id]
        context = {
            SOURCE_FEATURE: card["Адрес источника"], GROUPING_FEATURE: str(card["Элементов группировки"]),
            "Наименование": card["rule"], "Критичность": card["criticality"],
            "Пиктограмма IRP": card[CATEGORY],
        }
        return pd.DataFrame({
            "Признак": values.index, "Вклад": values.values, "Значение": values.index.map(context).fillna(""),
        })

    def by_outcome(self, cards: pd.DataFrame) -> pd.DataFrame:
        actual = cards.loc[cards["y"].eq(1)]
        means = self.values.loc[actual.index].groupby(actual["outcome"]).mean().reindex(["FN", "TP"])
        # Сравниваем только исходы, представленные в группе.
        means = means.dropna(how="all").T
        order = means.abs().max(axis=1).nlargest(8).index
        return means.loc[order]


def source_kind(addresses: list[str], is_private) -> str:
    kinds = {is_private(ip) for ip in addresses}
    if not kinds:
        return "Не указан"
    if len(kinds) > 1:
        return "Смешанный"
    return "Внутренний" if True in kinds else "Внешний"
