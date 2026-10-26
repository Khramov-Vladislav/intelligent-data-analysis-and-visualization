"""СРС 2.2, артефакт «после»: топ-8 правил SIEM по числу карточек (выборка 1000 карточек).

Запуск: python srs22_after_chart.py [путь_к_parquet] [выходной_png]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lecture/Dataset/iavd_1000_incidents_raw_target.parquet"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "DV/img/srs22_after.png"

counts = pd.read_parquet(SRC, columns=["Наименование"])["Наименование"].value_counts().head(8)[::-1]

fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
colors = ["#C0392B" if i == len(counts) - 1 else "#B8BEC6" for i in range(len(counts))]
ax.barh(counts.index, counts.values, color=colors, height=0.65)
for y, v in enumerate(counts.values):
    ax.text(v + 1.5, y, str(v), va="center", fontsize=11)
fig.suptitle("Алерты по правилам SIEM: топ-8 правил, число карточек (выборка 1000 карточек)",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
ax.tick_params(axis="y", labelsize=10, length=0)
ax.xaxis.set_visible(False)
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)
ax.set_xlim(0, counts.max() * 1.12)
fig.tight_layout()
fig.savefig(OUT, facecolor="white")
print(f"saved {OUT}")
