"""Показатели СРС 2.4 и обзора потока SOC."""

import pandas as pd

from data import CRIT_ORDER, FP, NO_VERDICT, RULE

MIN_RULE_CARDS = 300
MIN_RULE_FP_SHARE = 0.95
HOUR_LOAD_LIMITS = [0, 5, 10, 20, 50, 100, float("inf")]
HOUR_LOAD_LABELS = ["1–5", "6–10", "11–20", "21–50", "51–100", ">100"]


def rule_noise(cards: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Доля FP по правилам среди карточек с содержательным вердиктом."""
    analyzed = cards["Вердикт"].notna() & ~cards["Вердикт"].isin([NO_VERDICT, "Дубликат"])
    reviewed = cards.loc[analyzed, [RULE, "Вердикт"]].dropna(subset=[RULE])
    fp = reviewed["Вердикт"].eq(FP)
    rules = fp.groupby(reviewed[RULE]).agg(cards="size", false_positives="sum")
    rules["fp_share"] = rules["false_positives"] / rules["cards"]
    rules["noisy"] = rules["cards"].ge(MIN_RULE_CARDS) & rules["fp_share"].ge(MIN_RULE_FP_SHARE)
    return rules, int(analyzed.sum())


def take_sla_by_load(cards: pd.DataFrame) -> pd.DataFrame:
    """Доля просрочек среди карточек с известным флагом просрочки."""
    bucket = pd.cut(cards["Карточек в часе"], HOUR_LOAD_LIMITS, labels=HOUR_LOAD_LABELS)
    return cards["Флаг просрочки взятия"].groupby(bucket, observed=False).agg(
        cards="count", overdue_share="mean"
    )


def take_time_by_criticality(cards: pd.DataFrame) -> pd.DataFrame:
    """Время реакции и доля карточек, взятых позже индивидуального норматива."""
    observed = cards.loc[cards["До взятия, мин"].notna()]
    minutes = observed["До взятия, мин"]
    sla = observed["SLA взятия, мин"]
    overtime = minutes.gt(sla).where(sla.notna())
    severity = observed["Критичность"]
    times = minutes.groupby(severity, observed=True)
    quartiles = times.quantile([0.25, 0.5, 0.75]).unstack().reindex(columns=[0.25, 0.5, 0.75])
    quartiles = quartiles.rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
    breaches = overtime.groupby(severity, observed=True).agg(with_sla="count", overdue_share="mean")
    return quartiles.join(times.size().rename("cards")).join(breaches).reindex(CRIT_ORDER)
