"""
make_assets.py  -  builds every number, table and figure used in report/main.tex
from the REAL Raisin and Abalone data.

Run it from the root of the repository:
    python report/make_assets.py

It creates:
    report/results.tex      (all numbers, as LaTeX macros)
    report/tables/*.tex     (all tables)
    report/figures/*.png    (all figures)
    data/processed/*.csv    (cleaned / encoded data)
"""
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
for p in (RAW, PROC, FIG, TAB):
    os.makedirs(p, exist_ok=True)

sns.set_theme(style="whitegrid")
R = {}   # macro name -> text


# ------------------------------------------------------------------ helpers
def esc(s):
    return str(s).replace("_", r"\_")

def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

def fmt(x):
    """Format one number for a table."""
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-4:
        return sci(x)
    if abs(x) >= 100:
        return f"{x:,.1f}"
    return f"{x:.4g}"

def pct(x, k=1):
    return f"{100 * x:.{k}f}\\%"

def tex_table(header, rows, align):
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MIDRULE":
            lines.append("\\midrule")
            continue
        cells = [re.sub(r"(?<![\w$.\\{])-(?=\d)", "$-$", str(c)) for c in r]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"

def save_table(name, header, rows, align):
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        f.write(tex_table(header, rows, align))

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close()

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    i = q3 - q1
    return q1 - k * i, q3 + k * i

def load(name, uci_id):
    """Read the data. Order: data/raw/<name>_raw.csv -> file downloaded by hand from UCI -> download."""
    path = os.path.join(RAW, f"{name}_raw.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # files downloaded by hand from the UCI website
    manual_xlsx = os.path.join(RAW, "Raisin_Dataset.xlsx")
    manual_csv = os.path.join(RAW, "Raisin_Dataset.csv")
    manual_data = os.path.join(RAW, "abalone.data")
    df = None
    if name == "raisin" and os.path.exists(manual_xlsx):
        df = pd.read_excel(manual_xlsx)
    elif name == "raisin" and os.path.exists(manual_csv):
        df = pd.read_csv(manual_csv)
    elif name == "abalone" and os.path.exists(manual_data):
        df = pd.read_csv(manual_data, header=None,
                         names=["Sex", "Length", "Diameter", "Height", "Whole_weight",
                                "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"])
    if df is None:
        try:
            from ucimlrepo import fetch_ucirepo
            d = fetch_ucirepo(id=uci_id)
            df = pd.concat([d.data.features, d.data.targets], axis=1)
        except Exception as e:
            print(f"\nCould not download {name} from UCI: {e}")
            print("Download it by hand from the UCI page, put the file in data/raw/ and run again.")
            print("  Raisin : data/raw/Raisin_Dataset.xlsx")
            print("  Abalone: data/raw/abalone.data")
            sys.exit(1)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------------ load
raisin = load("raisin", 850)
abalone = load("abalone", 1)
print("Raisin shape :", raisin.shape, "(UCI says 900 x 8)")
print("Abalone shape:", abalone.shape, "(UCI says 4177 x 9)")
if raisin.shape != (900, 8) or abalone.shape != (4177, 9):
    print("WARNING: the shape is different from UCI. Check that you use the UCI files.")

R["RaisinRowsRaw"] = f"{len(raisin):,}"
R["AbaloneRowsRaw"] = f"{len(abalone):,}"
R["RaisinColsRaw"] = str(raisin.shape[1])
R["AbaloneColsRaw"] = str(abalone.shape[1])

R_FEATS = [c for c in raisin.columns if c != "Class"]
A_FEATS = [c for c in abalone.columns if c not in ("Sex", "Rings")]

# short names for wide tables
SHORT = {"MajorAxisLength": "MajAx", "MinorAxisLength": "MinAx", "Eccentricity": "Ecc",
         "ConvexArea": "CvxA", "Perimeter": "Per", "Extent": "Ext",
         "Whole_weight": "WholeW", "Shucked_weight": "ShuckW",
         "Viscera_weight": "ViscW", "Shell_weight": "ShellW",
         "Diameter": "Diam", "Length": "Len", "Height": "Hgt"}
sh = lambda c: SHORT.get(c, c)


# ------------------------------------------------------------------ class balance
r_counts = raisin["Class"].value_counts()
R["RaisinClassA"] = esc(r_counts.index[0])
R["RaisinClassB"] = esc(r_counts.index[1])
R["RaisinClassANum"] = str(r_counts.iloc[0])
R["RaisinClassBNum"] = str(r_counts.iloc[1])

a_counts = abalone["Sex"].value_counts()
for s in ("M", "F", "I"):
    R[f"AbaloneSex{s}"] = str(int(a_counts.get(s, 0)))
R["AbaloneRingsMin"] = str(int(abalone["Rings"].min()))
R["AbaloneRingsMax"] = str(int(abalone["Rings"].max()))
R["AbaloneRingsMedian"] = f"{abalone['Rings'].median():.0f}"


# ------------------------------------------------------------------ quality checks (before cleaning)
def outlier_counts(df, cols):
    per = {}
    rows_with = pd.Series(False, index=df.index)
    for c in cols:
        lo, hi = iqr_bounds(df[c])
        mask = (df[c] < lo) | (df[c] > hi)
        per[c] = int(mask.sum())
        rows_with |= mask
    return per, int(rows_with.sum())

# Raisin
r_missing = int(raisin.isna().sum().sum())
r_dups = int(raisin.duplicated().sum())
r_nonpos = int((raisin[R_FEATS] <= 0).sum().sum())
r_incons = int((raisin["ConvexArea"] < raisin["Area"]).sum()
               + (raisin["MinorAxisLength"] > raisin["MajorAxisLength"]).sum()
               + ((raisin["Eccentricity"] < 0) | (raisin["Eccentricity"] > 1)).sum()
               + ((raisin["Extent"] < 0) | (raisin["Extent"] > 1)).sum())
r_out_per, r_out_rows = outlier_counts(raisin, R_FEATS)
r_out_total = sum(r_out_per.values())

# Abalone
a_missing = int(abalone.isna().sum().sum())
a_dups = int(abalone.duplicated().sum())
a_height0 = int((abalone["Height"] <= 0).sum())
a_nonpos_other = int((abalone[[c for c in A_FEATS if c != "Height"]] <= 0).sum().sum())
parts = abalone["Shucked_weight"] + abalone["Viscera_weight"] + abalone["Shell_weight"]
short_mask = abalone["Whole_weight"] < parts - 1e-9
a_whole_short = int(short_mask.sum())
a_diam_len = int((abalone["Diameter"] > abalone["Length"]).sum())
a_incons = a_whole_short + a_diam_len
a_sex_values = sorted(abalone["Sex"].dropna().unique().tolist())
a_out_per, a_out_rows = outlier_counts(abalone, A_FEATS)
a_out_total = sum(a_out_per.values())

R.update({
    "RaisinMissing": str(r_missing), "AbaloneMissing": str(a_missing),
    "RaisinDups": str(r_dups), "AbaloneDups": str(a_dups),
    "RaisinNonPos": str(r_nonpos), "AbaloneNonPos": str(a_height0 + a_nonpos_other),
    "AbaloneHeightZero": str(a_height0),
    "RaisinIncons": str(r_incons), "AbaloneIncons": str(a_incons),
    "AbaloneWholeShort": str(a_whole_short), "AbaloneDiamLen": str(a_diam_len),
    "AbaloneWholeShortPct": pct(a_whole_short / len(abalone)),
    "AbaloneSexValues": ", ".join(a_sex_values),
    "RaisinOutTotal": str(r_out_total), "AbaloneOutTotal": str(a_out_total),
    "RaisinOutRows": str(r_out_rows), "AbaloneOutRows": str(a_out_rows),
    "RaisinOutRowsPct": pct(r_out_rows / len(raisin)),
    "AbaloneOutRowsPct": pct(a_out_rows / len(abalone)),
})
worst_r = max(r_out_per, key=r_out_per.get)
worst_a = max(a_out_per, key=a_out_per.get)
R["RaisinOutWorst"] = esc(worst_r)
R["RaisinOutWorstNum"] = str(r_out_per[worst_r])
R["AbaloneOutWorst"] = esc(worst_a)
R["AbaloneOutWorstNum"] = str(a_out_per[worst_a])

save_table("quality",
           ["Check", "Raisin", "Abalone"],
           [["Missing values", r_missing, a_missing],
            ["Duplicate rows", r_dups, a_dups],
            ["Non-positive values in the features", r_nonpos, a_height0 + a_nonpos_other],
            ["Rows that break a physical rule", r_incons, a_incons],
            ["Outlier values (IQR rule)", r_out_total, a_out_total],
            ["Rows with at least one outlier", r_out_rows, a_out_rows]],
           "lrr")

save_table("outliers_per_feature",
           ["Feature", "Outliers"],
           [[esc(c), r_out_per[c]] for c in R_FEATS],
           "lr")
save_table("outliers_per_feature_ab",
           ["Feature", "Outliers"],
           [[esc(c), a_out_per[c]] for c in A_FEATS],
           "lr")


# ------------------------------------------------------------------ cleaning
def clean(df, feats, drop_height=False):
    info = {}
    d = df.copy()
    info["before"] = len(d)
    # missing
    n = len(d); d = d.dropna(); info["missing_rows"] = n - len(d)
    # duplicates
    n = len(d); d = d.drop_duplicates(); info["dup_rows"] = n - len(d)
    # invalid values
    n = len(d)
    if drop_height:
        d = d[d["Height"] > 0]
    info["invalid_rows"] = n - len(d)
    d = d.copy()
    # clip outliers (IQR rule)
    clipped = {}
    for c in feats:
        lo, hi = iqr_bounds(d[c])
        clipped[c] = int(((d[c] < lo) | (d[c] > hi)).sum())
        d[c] = d[c].clip(lo, hi)
    info["clipped"] = clipped
    info["clipped_total"] = sum(clipped.values())
    info["after"] = len(d)
    return d, info

raisin_c, ri = clean(raisin, R_FEATS)
abalone_c, ai = clean(abalone, A_FEATS, drop_height=True)

for tag, info in (("Raisin", ri), ("Abalone", ai)):
    R[f"{tag}RowsAfter"] = f"{info['after']:,}"
    R[f"{tag}MissingRows"] = str(info["missing_rows"])
    R[f"{tag}DupRows"] = str(info["dup_rows"])
    R[f"{tag}InvalidRows"] = str(info["invalid_rows"])
    R[f"{tag}Clipped"] = str(info["clipped_total"])

save_table("cleaning_summary",
           ["Dataset", "Rows before", "Missing", "Duplicates", "Invalid", "Rows after", "Clipped values"],
           [["Raisin", ri["before"], ri["missing_rows"], ri["dup_rows"], ri["invalid_rows"], ri["after"], ri["clipped_total"]],
            ["Abalone", ai["before"], ai["missing_rows"], ai["dup_rows"], ai["invalid_rows"], ai["after"], ai["clipped_total"]]],
           "lrrrrrr")

save_table("quality",
           ["Check", "Raisin", "Abalone"],
           [["Missing values", r_missing, a_missing],
            ["Duplicate rows", r_dups, a_dups],
            ["Non-positive values in the features", r_nonpos, a_height0 + a_nonpos_other],
            ["Rows that break a physical rule", r_incons, a_incons],
            ["Outlier values (IQR rule)", r_out_total, a_out_total],
            ["Rows with at least one outlier", r_out_rows, a_out_rows],
            "MIDRULE",
            ["Rows removed by the cleaning", ri["before"] - ri["after"], ai["before"] - ai["after"]],
            ["Values clipped", ri["clipped_total"], ai["clipped_total"]],
            ["Rows after cleaning", ri["after"], ai["after"]]],
           "lrr")

def before_after_rows(df_b, df_a, feats):
    rows = [[esc(c), fmt(df_b[c].mean()), fmt(df_a[c].mean()),
             fmt(df_b[c].std()), fmt(df_a[c].std())] for c in feats]
    drops = {c: 1 - df_a[c].std() / df_b[c].std() for c in feats}
    best = max(drops, key=drops.get)
    return rows, best, drops[best]

rows_r, b, dr = before_after_rows(raisin, raisin_c, R_FEATS)
R["RaisinStdDropFeat"] = esc(b); R["RaisinStdDropPct"] = pct(dr)
rows_a, b, dr = before_after_rows(abalone, abalone_c, A_FEATS)
R["AbaloneStdDropFeat"] = esc(b); R["AbaloneStdDropPct"] = pct(dr)
save_table("before_after", ["Feature", "Mean before", "Mean after", "Std before", "Std after"],
           [["\\multicolumn{5}{l}{\\textit{Raisin}}"]] + rows_r +
           [["\\multicolumn{5}{l}{\\textit{Abalone}}"]] + rows_a, "lrrrr")

# class / sex / rings numbers from the cleaned data (so text and figures agree)
r_counts = raisin_c["Class"].value_counts()
R["RaisinClassA"] = esc(r_counts.index[0]); R["RaisinClassB"] = esc(r_counts.index[1])
R["RaisinClassANum"] = str(r_counts.iloc[0]); R["RaisinClassBNum"] = str(r_counts.iloc[1])
a_counts = abalone_c["Sex"].value_counts()
for s_ in ("M", "F", "I"):
    R[f"AbaloneSex{s_}"] = str(int(a_counts.get(s_, 0)))
R["AbaloneRingsMin"] = str(int(abalone_c["Rings"].min()))
R["AbaloneRingsMax"] = str(int(abalone_c["Rings"].max()))
R["AbaloneRingsMedian"] = f"{abalone_c['Rings'].median():.0f}"

# save cleaned data
raisin_c.to_csv(os.path.join(PROC, "raisin_clean.csv"), index=False)
abalone_c.to_csv(os.path.join(PROC, "abalone_clean.csv"), index=False)
pd.get_dummies(abalone_c, columns=["Sex"], dtype=int).to_csv(
    os.path.join(PROC, "abalone_encoded.csv"), index=False)

# figure 1: boxplot of Abalone Height before and after cleaning
height_before = pd.DataFrame({"Height": abalone["Height"], "Data": "Before cleaning"})
height_after = pd.DataFrame({"Height": abalone_c["Height"], "Data": "After cleaning"})
height_all = pd.concat([height_before, height_after])

plt.figure(figsize=(5.5, 4.5))
sns.boxplot(data=height_all, x="Data", y="Height", color="lightblue")
plt.title("Abalone: Height before and after cleaning")
plt.xlabel("")
plt.ylabel("Height")
save_fig("fig_box")

# ------------------------------------------------------------------ statistics
def stats_table(df, feats, name):
    rows = [[esc(c), fmt(df[c].mean()), fmt(df[c].var()), fmt(df[c].std())] for c in feats]
    save_table(name, ["Feature", "Mean", "Variance", "Std"], rows, "lrrr")

stats_table(raisin_c, R_FEATS, "stats_raisin")
stats_table(abalone_c, A_FEATS, "stats_abalone")

def cov_table(df, feats, name):
    cov = df[feats].cov()
    rows = [[esc(r)] + [fmt(cov.loc[r, c]) for c in feats] for r in feats]
    save_table(name, ["Feature"] + [sh(c) for c in feats], rows, "l" + "r" * len(feats))

cov_table(raisin_c, R_FEATS, "cov_raisin")
cov_table(abalone_c, A_FEATS, "cov_abalone")

def strongest(df, feats):
    corr = df[feats].corr()
    best, val = None, 0
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if abs(corr.loc[a, b]) > abs(val):
                best, val = (a, b), corr.loc[a, b]
    return corr, best, val

for tag, df, feats in (("Raisin", raisin_c, R_FEATS), ("Abalone", abalone_c, A_FEATS)):
    corr, best, val = strongest(df, feats)
    R[f"{tag}PairA"] = esc(best[0]); R[f"{tag}PairB"] = esc(best[1])
    R[f"{tag}PairR"] = f"{val:.3f}"
    off = corr.values[np.triu_indices(len(feats), 1)]
    R[f"{tag}CorrMin"] = f"${off.min():.2f}$"
    R[f"{tag}CorrMax"] = f"${off.max():.2f}$"
    R[f"{tag}CorrStrong"] = str(int((np.abs(off) > 0.9).sum()))
    R[f"{tag}CorrPairs"] = str(len(off))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for a, df, feats, title in ((ax[0], raisin_c, R_FEATS, "Raisin"), (ax[1], abalone_c, A_FEATS, "Abalone")):
    c = df[feats].corr()
    c.index = [sh(x) for x in c.index]; c.columns = [sh(x) for x in c.columns]
    sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                annot_kws={"size": 8}, cbar=False, ax=a)
    a.set_title(f"{title}: correlation matrix")
save_fig("fig_corr")

# categorical analysis
# Raisin: mean of each feature in each class
cls = sorted(raisin_c["Class"].unique())
rows = []
for c in R_FEATS:
    rows.append([esc(c)] + [fmt(raisin_c.loc[raisin_c["Class"] == k, c].mean()) for k in cls])
save_table("raisin_by_class", ["Feature"] + [esc(k) for k in cls], rows, "lrr")

# Abalone: Sex vs age group
age_group = pd.cut(abalone_c["Rings"], bins=[0, 8, 10, 40], labels=["young", "middle", "old"])
ct = pd.crosstab(abalone_c["Sex"], age_group)
chi2, p_chi, _, _ = chi2_contingency(ct)
n_obs = ct.values.sum()
cramer = np.sqrt(chi2 / (n_obs * (min(ct.shape) - 1)))
groups = [g["Rings"].values for _, g in abalone_c.groupby("Sex")]
f_stat, p_anova = f_oneway(*groups)
R["AbaloneChiP"] = "$<0.001$" if p_chi < 0.001 else f"$={p_chi:.3f}$"
R["AbaloneAnovaP"] = "$<0.001$" if p_anova < 0.001 else f"$={p_anova:.3f}$"
R["AbaloneCramer"] = f"{cramer:.3f}"
R["AbaloneSexWord"] = "is" if p_chi < 0.05 else "is not"
R["AbaloneCramerWord"] = ("weak" if cramer < 0.2 else "moderate" if cramer < 0.4 else "strong")
mean_rings = abalone_c.groupby("Sex")["Rings"].mean()
R["AbaloneLowRingsSex"] = mean_rings.idxmin()
R["AbaloneLowRingsMean"] = f"{mean_rings.min():.2f}"
R["AbaloneHighRingsSex"] = mean_rings.idxmax()
R["AbaloneHighRingsMean"] = f"{mean_rings.max():.2f}"
save_table("abalone_sex",
           ["Sex", "Samples", "Mean Rings"] + [f"{g}" for g in ct.columns],
           [[s, int((abalone_c["Sex"] == s).sum()), f"{mean_rings[s]:.2f}"] + [int(ct.loc[s, g]) for g in ct.columns]
            for s in ct.index],
           "lrr" + "r" * len(ct.columns))


# ------------------------------------------------------------------ PCA
def run_pca(df, feats):
    Xs = StandardScaler().fit_transform(df[feats])
    pca = PCA().fit(Xs)
    Z = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    mse = []
    for k in range(1, len(feats) + 1):
        p = PCA(n_components=k).fit(Xs)
        back = p.inverse_transform(p.transform(Xs))
        mse.append(float(np.mean((Xs - back) ** 2)))
    return Xs, pca, Z, evr, cum, mse

Xr, pr, Zr, evr_r, cum_r, mse_r = run_pca(raisin_c, R_FEATS)
Xa, pa, Za, evr_a, cum_a, mse_a = run_pca(abalone_c, A_FEATS)

for tag, evr, cum, pca in (("Raisin", evr_r, cum_r, pr), ("Abalone", evr_a, cum_a, pa)):
    R[f"{tag}PCOne"] = pct(evr[0])
    R[f"{tag}PCTwo"] = pct(evr[1])
    R[f"{tag}PCLast"] = pct(evr[-1], 2)
    R[f"{tag}CumTwo"] = pct(cum[1])
    R[f"{tag}CumThree"] = pct(cum[2])
    R[f"{tag}KEighty"] = str(int(np.argmax(cum >= 0.80) + 1))
    R[f"{tag}KNinety"] = str(int(np.argmax(cum >= 0.90) + 1))
    R[f"{tag}KNinetyFive"] = str(int(np.argmax(cum >= 0.95) + 1))
    R[f"{tag}Eig"] = f"{pca.explained_variance_[0]:.2f}"
    R[f"{tag}NumPC"] = str(len(evr))
    first = pca.components_[0]
    R[f"{tag}PCOneSign"] = "the same sign" if (np.all(first > 0) or np.all(first < 0)) else "mixed signs"

save_table("pca_table",
           ["PC / k", "Raisin ratio", "Raisin kept", "Abalone ratio", "Abalone kept"],
           [[f"{i+1}", pct(evr_r[i], 2), pct(cum_r[i], 2), pct(evr_a[i], 2), pct(cum_a[i], 2)]
            for i in range(len(evr_r))],
           "lrrrr")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, evr, cum, title in ((ax[0], evr_r, cum_r, "Raisin"), (ax[1], evr_a, cum_a, "Abalone")):
    x = np.arange(1, len(evr) + 1)
    a.bar(x, evr, color="#4c72b0", label="Each PC")
    a.plot(x, cum, marker="o", color="#c44e52", label="Cumulative")
    a.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    a.set_title(f"{title}: explained variance"); a.set_xlabel("Principal component")
    a.set_ylim(0, 1.05)
    a.legend(loc="center right")
ax[0].set_ylabel("Explained variance ratio")
save_fig("fig_scree")

def loadings_table(pca, feats, name):
    L = pca.components_[:2].T
    save_table(name, ["Feature", "PC1", "PC2"],
               [[esc(f), f"{L[i, 0]:.3f}", f"{L[i, 1]:.3f}"] for i, f in enumerate(feats)], "lrr")

loadings_table(pr, R_FEATS, "loadings_raisin")
loadings_table(pa, A_FEATS, "loadings_abalone")

# 2D plots
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for k in cls:
    m = (raisin_c["Class"] == k).values
    ax.scatter(Zr[m, 0], Zr[m, 1], s=14, alpha=0.6, label=k)
ax.set_xlabel(f"PC1 ({evr_r[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr_r[1]:.1%})")
ax.set_title("Raisin: PC1 vs PC2"); ax.legend(title="Class")
save_fig("fig_pca_raisin")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
sc = ax[0].scatter(Za[:, 0], Za[:, 1], c=abalone_c["Rings"], cmap="viridis", s=8, alpha=0.6)
plt.colorbar(sc, ax=ax[0], label="Rings")
ax[0].set_title("Abalone: colored by Rings")
for s_ in ("I", "F", "M"):
    m = (abalone_c["Sex"] == s_).values
    ax[1].scatter(Za[m, 0], Za[m, 1], s=8, alpha=0.4, label=s_)
ax[1].legend(title="Sex"); ax[1].set_title("Abalone: colored by Sex")
for a in ax:
    a.set_xlabel(f"PC1 ({evr_a[0]:.1%})"); a.set_ylabel(f"PC2 ({evr_a[1]:.1%})")
save_fig("fig_pca_abalone")

# how much label information do 2 PCs keep?
y_r = (raisin_c["Class"] == cls[0]).astype(int).values
cv = StratifiedKFold(5, shuffle=True, random_state=42)
acc2 = cross_val_score(make_pipeline(StandardScaler(), PCA(2), LogisticRegression(max_iter=1000)),
                       raisin_c[R_FEATS], y_r, cv=cv).mean()
accall = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                         raisin_c[R_FEATS], y_r, cv=cv).mean()
R["RaisinAccTwo"] = pct(acc2); R["RaisinAccAll"] = pct(accall)
diff = accall - acc2
R["RaisinKeepPhrase"] = "almost all" if diff <= 0.02 else ("a large part of" if diff <= 0.06 else "only part of")

rho1 = abs(spearmanr(Za[:, 0], abalone_c["Rings"])[0])
rho2 = abs(spearmanr(Za[:, 1], abalone_c["Rings"])[0])
R["AbaloneRhoOne"] = f"{rho1:.2f}"; R["AbaloneRhoTwo"] = f"{rho2:.2f}"
R["AbaloneRhoWord"] = "strong" if rho1 >= 0.6 else ("moderate" if rho1 >= 0.3 else "weak")


# ------------------------------------------------------------------ write macros
with open(os.path.join(HERE, "results.tex"), "w") as f:
    f.write("% Generated by make_assets.py - do not edit by hand\n")
    for k, v in R.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
print(f"Done. {len(R)} numbers written to report/results.tex, tables in report/tables, figures in report/figures.")