"""Построить графики СРС 3.3 и сверить расчёт для карточки 527528."""
import argparse
import json
from math import exp, log
from pathlib import Path
from zipfile import ZipFile

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE = "create_lag_s"
CARD_ID = 527528
FIGURE = Path(__file__).resolve().parents[1] / "img" / "srs33_pdp_ice_ale.png"


def read_csv(archive: ZipFile, name: str) -> pd.DataFrame:
    with archive.open(name) as source:
        return pd.read_csv(source, sep=";", encoding="utf-8-sig")


def plot_curves(effects: pd.DataFrame, ice: pd.DataFrame, destination: Path) -> None:
    fig, (average, individual) = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained")
    fig.set_facecolor("white")
    for ax in (average, individual):
        ax.set_xticks([0, 25, 50, 75, 90, 100])
        ax.set_xlim(-2, 102)
        ax.set_xlabel("Процентиль задержки", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b7bec6")

    average.set_title("Сдвиг относительно минимума", loc="left", fontsize=11)
    average.set_ylabel("Изменение оценки, п. п.", fontsize=10)
    for column, label, color in (("mean_proba", "PDP", "#2157a0"), ("ale", "ALE", "#9b4f10")):
        values = effects[column]
        average.plot(effects.index * 100, (values - values.iloc[0]) * 100,
                     color=color, marker="o", markersize=3, label=label)
    average.legend(frameon=False, loc="lower left", fontsize=9)

    individual.set_title("ICE для 20 карточек", loc="left", fontsize=11)
    individual.set_ylabel("Оценка инцидента, %", fontsize=10)
    for _, scores in ice.iterrows():
        high = scores.max() >= 0.5
        individual.plot(ice.columns * 100, scores * 100,
                        color="#2157a0" if high else "#9ca7b4", lw=1.7 if high else 0.8)
    individual.set_ylim(-3, 103)
    fig.savefig(destination, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Архив 3.3_XAI_данные.zip")
    parser.add_argument("shap_cards", type=Path, help="Файл course_pack/shap_cards.csv из комплекта ML 3.7")
    args = parser.parse_args()

    with ZipFile(args.archive) as archive:
        pdp = read_csv(archive, "pdp_numeric.csv").query("feature == @FEATURE").set_index("quantile")
        ale = read_csv(archive, "ale_numeric.csv").query("feature == @FEATURE").set_index("quantile")
        effects = pdp[["raw_value", "mean_proba"]].join(ale[["ale"]], how="inner").sort_index()
        ice_rows = read_csv(archive, "ice_numeric.csv").query("feature == @FEATURE")
        ice = ice_rows.pivot(index="line", columns="quantile", values="proba")
        importance = read_csv(archive, "importance.csv").set_index("source_column")
        ablation = read_csv(archive, "ablation.csv").set_index("field")
        cards = read_csv(archive, "cards_practice.csv").set_index("EntityId")
        threshold = json.loads(archive.read("spec.json"))["threshold"]

    ranks = importance[["rank", "mean_abs_shap"]].join(
        ablation[["rank", "mean_abs_delta"]], lsuffix="_shap", rsuffix="_ablation",
    )
    top = ranks.loc[ranks["rank_shap"].le(5) | ranks["rank_ablation"].le(5)]
    print(effects.to_string())
    print(ice[[0.0, 0.9, 1.0]].to_string())
    print(top.to_string())
    print(f"Корреляция рангов: {ranks['rank_shap'].corr(ranks['rank_ablation']):.6f}")

    card = cards.loc[CARD_ID]
    shap = pd.read_csv(args.shap_cards, sep=";", encoding="utf-8-sig").set_index("EntityId")
    row = shap.loc[CARD_ID]
    contributions = row.drop(["base", "y_true", "proba", "outcome"]).astype(float)
    z_shap = row["base"] + contributions.sum()
    p_shap = 1 / (1 + exp(-z_shap))
    p = card["proba"]
    z_csv = log(p / (1 - p))
    z_threshold = log(threshold / (1 - threshold))
    if abs(p_shap - p) > 5e-7 or (z_shap >= z_threshold) != (p >= threshold):
        raise ValueError("Сумма вкладов не воспроизводит оценку или решение из CSV.")

    print(f"Карточка {CARD_ID}, набор {card['team']}, исход {card['outcome']}, p = {p:.6f}")
    print(f"z из CSV = {z_csv:.10f}, порог = {z_threshold:.10f}")
    print(f"База {row['base']:.6f} + сумма {contributions.sum():.10f} = {z_shap:.10f}")
    print(f"p из SHAP = {p_shap:.10f}, разница {abs(p_shap - p):.2e}")
    print(contributions.reindex(contributions.abs().nlargest(5).index).to_string())
    plot_curves(effects, ice, FIGURE)


if __name__ == "__main__":
    main()
