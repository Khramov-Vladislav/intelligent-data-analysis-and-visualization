"""СРС 3.2. Сравнение порогов на размеченной части учебного лога."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ML" / "app"))

from inference_metrics import threshold_counts

LOG = "MODEL_PREDICTIONS_target.parquet"
COURSE_THRESHOLD = 0.18
SCENARIOS = (50, 75, 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help=f"Путь к {LOG}")
    args = parser.parse_args()
    log = pd.read_parquet(
        args.log,
        columns=["Дата", "version", "Flag_ValidLabel", "y", "y_pred_proba", "y_pred", "confusion"],
    )
    valid = log.loc[log["Flag_ValidLabel"].eq(1)]
    y, scores = valid["y"], valid["y_pred_proba"]
    if set(y.unique()) != {0, 1} or not scores.between(0, 1).all():
        raise ValueError("В размеченном срезе нужны оба класса и оценки от 0 до 1.")
    if valid["version"].nunique() != 1:
        raise ValueError("Для сравнения порогов нужен лог одной версии модели.")

    curve = threshold_counts(scores, y)[["TP", "FP", "FN", "TN"]]
    course_pred = scores.ge(COURSE_THRESHOLD)
    if not course_pred.eq(valid["y_pred"]).all():
        raise ValueError("Решения при пороге курса расходятся с y_pred.")
    course_row = curve.loc[curve.index >= COURSE_THRESHOLD].iloc[-1]
    if course_row.to_dict() != valid["confusion"].value_counts().to_dict():
        raise ValueError("Матрица при пороге курса расходится с полем confusion.")

    print(f"Модель {valid['version'].iloc[0]}, даты {log['Дата'].min()} – {log['Дата'].max()}")
    print(f"Всего {len(log)}, размечено {len(valid)}, инцидентов {int(y.sum())}")
    print(f"Порог курса {COURSE_THRESHOLD}, {course_row.to_dict()}")
    print(f"Различных оценок {scores.nunique()}")
    curve["precision"] = curve["TP"] / (curve["TP"] + curve["FP"]).clip(lower=1)
    curve["recall"] = curve["TP"] / y.sum()
    for k in SCENARIOS:
        curve["cost"] = k * curve["FN"] + curve["FP"]
        numerator = (1 + k) * curve["TP"]
        curve["F_beta"] = numerator / (numerator + k * curve["FN"] + curve["FP"])
        # idxmin / idxmax берут первый порог при равенстве; пороги идут сверху вниз.
        t_cost = curve["cost"].idxmin()
        t_fbeta = curve["F_beta"].idxmax()
        course = curve.loc[curve.index >= COURSE_THRESHOLD].iloc[-1].rename(COURSE_THRESHOLD)
        results = pd.DataFrame([course, curve.loc[t_cost], curve.loc[t_fbeta]])
        results.index.name = "threshold"
        print(f"\nk={k}, β={np.sqrt(k):.4f} (порог курса, минимум стоимости, максимум Fβ)")
        print(results.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
