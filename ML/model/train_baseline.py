"""Обучение базовой модели-советчика (замена предобученной модели курса, пока ее нет).

Запуск: cd ML && poetry run python model/train_baseline.py
Результат: model/advisor_baseline.joblib + model/advisor_baseline.json (метаданные).

Основная модель: стратифицированное случайное разбиение 80/20. Дополнительно — проверка
по времени (обучение на ранних 80 % карточек, оценка на последних 20 %): доля инцидентов
по месяцам меняется от 3 % до 28 %, и эта проверка показывает, насколько модель
чувствительна к дрейфу. Ее результат сохраняется в метаданных как предупреждение.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from advisor import (CATEGORICAL, DEFAULT_MODEL, NUMERIC, RAW_COLUMNS, TEST_IDS,  # noqa: E402
                     build_features, labels)
from data import dataset_path  # noqa: E402

TEST_SHARE = 0.2
MIN_RECALL = 0.90  # нижняя граница полноты из СРС к 3.2


def make_pipeline() -> Pipeline:
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1, min_frequency=20, max_categories=250), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    clf = HistGradientBoostingClassifier(
        categorical_features=CATEGORICAL,
        learning_rate=0.08,
        max_iter=400,
        max_leaf_nodes=48,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline([("encode", encode), ("clf", clf)])


def pick_threshold(y: np.ndarray, p: np.ndarray, min_recall: float) -> float:
    """Самый высокий порог, при котором полнота еще не ниже заданной: максимум точности при гарантии полноты."""
    _, recall, thresholds = precision_recall_curve(y, p)
    ok = thresholds[recall[:-1] >= min_recall]
    return float(np.floor(ok.max() * 1000) / 1000) if len(ok) else float(thresholds.min())


def temporal_check(X: pd.DataFrame, y: pd.Series, created: pd.Series) -> dict:
    """Оценка на последних по времени карточках: насколько модель переносится на будущий поток."""
    order = created.sort_values().index
    X, y = X.loc[order], y.loc[order]
    cut = int(len(X) * (1 - TEST_SHARE))
    p = make_pipeline().fit(X.iloc[:cut], y.iloc[:cut]).predict_proba(X.iloc[cut:])[:, 1]
    return {
        "test_from": created.loc[order].iloc[cut].isoformat(),
        "roc_auc": round(roc_auc_score(y.iloc[cut:], p), 4),
        "average_precision": round(average_precision_score(y.iloc[cut:], p), 4),
        "test_positive_share": round(float(y.iloc[cut:].mean()), 4),
    }


def main():
    raw = pd.read_parquet(dataset_path(), columns=RAW_COLUMNS)
    y = labels(raw)
    raw, y = raw[y.notna()], y[y.notna()].astype(int)
    X = build_features(raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SHARE, stratify=y, random_state=42)
    model = make_pipeline().fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    t = pick_threshold(y_test.values, p, MIN_RECALL)

    meta = {
        "name": "Базовая модель-советчик (HistGradientBoosting)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "positive_class": "реальный инцидент",
        "features": CATEGORICAL + NUMERIC,
        "split": "стратифицированное случайное 80/20, random_state=42",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_positive_share": round(float(y_test.mean()), 4),
        "threshold": t,
        "min_recall": MIN_RECALL,
        "metrics_test": {
            "roc_auc": round(roc_auc_score(y_test, p), 4),
            "average_precision": round(average_precision_score(y_test, p), 4),
            "precision_at_threshold": round(precision_score(y_test, p >= t), 4),
            "recall_at_threshold": round(recall_score(y_test, p >= t), 4),
        },
    }
    meta["temporal_check"] = temporal_check(X, y, raw["Создан"])
    joblib.dump(model, DEFAULT_MODEL, compress=3)
    np.save(TEST_IDS, raw.loc[X_test.index, "EntityId"].to_numpy())
    DEFAULT_MODEL.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
