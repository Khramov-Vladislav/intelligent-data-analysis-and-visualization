"""EDA-банк SOC (Python). Запуск: python eda_answers.py [путь_к_parquet]."""
import sys, pandas as pd
P = sys.argv[1] if len(sys.argv) > 1 else "iavd_incidents_raw_target.parquet"
df = pd.read_parquet(P)
SEP = "~~||~~"
def q(n, v): print(f"\n[{n}]\n{v}")
def safe(n, f):
    try: q(n, f())
    except Exception as e: q(n, f"(не вычислено: {e})")
vc = lambda c, k=None: df[c].value_counts(dropna=False).head(k)
t = lambda c: pd.to_numeric(df[c], errors="coerce")
d = pd.to_datetime(df["Создан"], errors="coerce")
fp = df["Вердикт"].eq("False Positive")

# ---- Блок 1
q("1.1", df.shape)
q("1.2", df.dtypes.astype(str).str.replace(r"\[.*\]", "", regex=True).value_counts())
q("1.3", df.isna().mean().nlargest(3))
q("1.4", (df.isna().mean() == 0).sum())
q("1.5", df["Вердикт"].nunique()); q("1.6", df["Критичность"].nunique())
c = "Значение Инцидента (группировка)"
q("1.7", df[c].dropna().astype(str).str.contains(r"\|", regex=True).mean())
q("1.8", df["Значение Инцидента"].dropna().astype(str).str.split(SEP, regex=False).str.len().max() if "Значение Инцидента" in df else df[c].dropna().astype(str).str.split(SEP, regex=False).str.len().max())
q("1.9", df["EntityId"].duplicated().sum())
q("1.10", df[c].dropna().astype(str).str.split(SEP, regex=False).str.len().sum())
q("1.11", "verdict")
# ---- Блок 2
q("2.1", vc("Вердикт")); q("2.2", fp.mean()); q("2.3", vc("Фаза реагирования"))
q("2.5", df["Фаза реагирования"].loc[lambda s: s.astype(str).str.contains(SEP, regex=False)].value_counts().head(1))
q("2.6", vc("Статус инцидента из SIEM")); q("2.7", vc("Спам")); q("2.8", vc("Эскалировался на L2?"))
q("2.9", vc("Проверка на FP выполнена")); q("2.10", pd.crosstab(df["Вердикт"], df["Ложное срабатывание"], dropna=False))
q("2.11", vc("Инцидент был изменен"))
q("2.12", df.groupby("Группа устранения")["Вердикт"].apply(lambda s: s.eq("False Positive").mean()))
# ---- Блок 3
for n, col in [("3.1","suser"),("3.2","Полное имя"),("3.5","Hostname Цели"),("3.7","Ответственный")]: q(n, df[col].nunique())
q("3.3", vc("Полное имя", 5)); q("3.4", vc("suser", 3)); q("3.6", vc("Hostname Цели", 3))
q("3.8", vc("Ответственный")); q("3.9", vc("Группа устранения")); q("3.10", vc("Конвейер SIEM"))
# ---- Блок 4
q("4.1", (d.min(), d.max())); m = d.dt.to_period("M").value_counts()
q("4.2", m.head(1)); q("4.3", m.iloc[:-1].tail(1) if False else m.sort_values().head(3))
h = d.dt.hour.value_counts(); q("4.4", (h.idxmax(), h.max(), h.idxmin(), h.min()))
q("4.5", d.dt.dayofweek.isin([5, 6]).mean())
tr = t("Время решения"); q("4.6", tr.median()/3600); q("4.7", tr.quantile(.9)/3600)
q("4.8", (tr > 30*86400).mean()); q("4.9", (tr.groupby(df["Группа устранения"]).median()/3600))
q("4.10", t("Период взятия в работу").median()/3600)
q("4.11", pd.to_datetime(t("задержка перед автоназначением").median(), unit="s"))
q("4.12", t("SLA по устранению Атаки").median()/3.6e6)
# ---- Блок 5
q("5.1", vc("processName", 8)); q("5.2", df["Команда"].astype(str).str.contains("powershell", case=False).mean())
q("5.3", vc("Пиктограмма IRP", 5)); q("5.4", vc("Теги", 3))
q("5.5", df["Теги"].dropna().astype(str).str.contains(SEP, regex=False).mean())
pr = (df[c].dropna().astype(str).str.split(SEP, regex=False).explode().str.split("|", n=1, expand=True))
safe("5.6", lambda: pr[1].value_counts()); safe("5.7", lambda: pr[0].value_counts().head(3))
safe("5.8", lambda: (len(pr), pr[0].nunique()))
q("5.9", df.assign(fp=fp).groupby("Критичность")["fp"].mean().sort_values(ascending=False))
q("5.10", df.assign(fp=fp).groupby("Группа устранения")["fp"].mean())
q("5.12", df[c].dropna().astype(str).str.split(SEP, regex=False).str.len().agg(["mean", "median"]))
# ---- Блок 6
q("6.2", df["Комментарий"].fillna("").eq("").sum()); q("6.3", df["Ошибки"].notna().sum())
q("6.4", df["Эскалировался на L2?"].isna().sum()); q("6.5", (df["Вердикт"].isna().sum(), (df["Вердикт"]=="Не указан").sum()))
q("6.6", df["Hostname Цели"].dropna().str.lower().nunique() - df["Hostname Цели"].nunique())
q("6.7", tr.max()/3600)
# ---- Блок 7
q("7.1", (fp.mean(), 1-fp.mean())); q("7.2", vc("Ложное срабатывание"))
q("7.3", df[["suser","duser","Hostname Цели","Наименование"]].nunique())
q("7.9", "train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)")
