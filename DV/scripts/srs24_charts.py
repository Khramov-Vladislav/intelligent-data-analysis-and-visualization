"""Пересчитать СРС 2.4 и сохранить три графика в DV/img."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ML" / "app"))

import plotly.io as pio

from charts import rule_noise_fig, take_sla_fig, take_time_fig
from data import load_cards
from soc_eda import rule_noise, take_sla_by_load, take_time_by_criticality


def main():
    cards = load_cards()
    rules, reviewed = rule_noise(cards)
    noisy = rules.loc[rules["noisy"]]
    load = take_sla_by_load(cards)
    response = take_time_by_criticality(cards)

    print(f"Карточек: {len(cards)}, с содержательным вердиктом: {reviewed}")
    print(f"Шумных правил: {len(noisy)} из {len(rules)}")
    print(f"Карточек по шумным правилам: {noisy['cards'].sum()}, из них FP: {noisy['false_positives'].sum()}")
    print(f"Без флага SLA: {cards['Флаг просрочки взятия'].isna().sum()}")
    print(load.to_string())
    print(f"Без корректной длительности: {cards['До взятия, мин'].isna().sum()}")
    print(response.to_string(), flush=True)

    figures = [rule_noise_fig(rules), take_sla_fig(load), take_time_fig(response)]
    names = ["srs24_rule_noise.png", "srs24_sla_by_load.png", "srs24_take_time.png"]
    pio.write_images(figures, [ROOT / "DV" / "img" / name for name in names], width=900, height=390, scale=1.5)


if __name__ == "__main__":
    main()
