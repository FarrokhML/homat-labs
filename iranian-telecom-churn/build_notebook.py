"""Builds analysis.ipynb from the cells below, so the notebook stays reviewable as plain Python.

    python build_notebook.py      # writes analysis.ipynb
    python run.py                 # executes it top to bottom (figures/ and results.json)
"""
import nbformat as nbf

CELLS = [
("md", """# ریزش مشتری در یک اپراتور مخابراتی ایرانی

داده: *Iranian Churn Dataset* (UCI، DOI: 10.24432/C5JW3Z، مجوز CC BY 4.0)؛ ۳٬۱۵۰ مشتری، تجمیع ۹ ماه اول، برچسب ریزش در ماه دوازدهم.
سؤال: اگر فقط بتوانیم به ۱۰٪ مشتری‌ها پیشنهاد نگه‌داشت بدهیم، کدام‌ها را انتخاب کنیم و این انتخاب نسبت به انتخاب تصادفی و یک قاعده‌ی ساده چقدر بهتر است؟"""),
("code", """import json, warnings
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
SEED = 1405
RESULTS = {}

for f in ("fonts/Vazirmatn-Regular.ttf", "fonts/Vazirmatn-Bold.ttf"):
    font_manager.fontManager.addfont(f)
matplotlib.rcParams.update({"font.family": "Vazirmatn", "axes.spines.top": False, "axes.spines.right": False,
                            "axes.edgecolor": "#9AA3AE", "axes.labelcolor": "#1F2937", "xtick.color": "#5B6675",
                            "ytick.color": "#5B6675", "figure.dpi": 110})
FA = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")
def fa(s):
    \"\"\"Persian digits. matplotlib 3.11 shapes and orders Persian text itself; reshaping it again reverses it.\"\"\"
    return str(s).translate(FA)
NAVY, CYAN, GRAY = "#1E40AF", "#0E7490", "#9AA3AE"
def save(fig, name, alt):
    fig.savefig(f"figures/{name}.png", bbox_inches="tight", facecolor="white")
    Image.open(f"figures/{name}.png").save(f"figures/{name}.webp", "WEBP", quality=90)
    RESULTS.setdefault("figures", {})[name] = alt
    plt.close(fig)"""),
("md", "## ۱. داده و پاک‌سازی"),
("code", """raw = pd.read_csv("data/customer-churn.csv")
raw.columns = [" ".join(c.split()) for c in raw.columns]      # "Call  Failure" -> "Call Failure"
dups = int(raw.duplicated().sum())
df = raw.drop_duplicates().reset_index(drop=True)
RESULTS["rows_raw"], RESULTS["duplicates"], RESULTS["rows"] = len(raw), dups, len(df)
RESULTS["churn_rate"] = round(float(df["Churn"].mean()) * 100, 1)
print(len(raw), dups, len(df), RESULTS["churn_rate"])"""),
("md", "**نکته:** ۳۰۰ ردیف کاملاً تکراری‌اند. اگر قبل از جدا کردن داده‌ی آموزش و آزمون حذف نشوند، نسخه‌ای از یک مشتری در هر دو طرف می‌افتد و مدل روی آزمون «جواب را از قبل دیده» است."),
("code", """# Age Group duplicates Age; Customer Value is a derived score. Keep Age, drop the group.
features = [c for c in df.columns if c not in ("Churn", "Age Group")]
X, y = df[features], df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
RESULTS["n_train"], RESULTS["n_test"], RESULTS["test_churners"] = len(X_train), len(X_test), int(y_test.sum())"""),
("md", "## ۲. چه کسانی می‌روند؟ (فقط روی داده‌ی آموزش)"),
("code", """tr = X_train.assign(Churn=y_train)
by_complain = tr.groupby("Complains")["Churn"].mean() * 100
use_bins = pd.qcut(tr["Seconds of Use"].rank(method="first"), 5, labels=False)
by_use = tr.groupby(use_bins)["Churn"].mean() * 100
by_status = tr.groupby("Status")["Churn"].mean() * 100
RESULTS["churn_complained"] = round(float(by_complain.loc[1]), 1)
RESULTS["churn_not_complained"] = round(float(by_complain.loc[0]), 1)
RESULTS["churn_lowest_use_quintile"] = round(float(by_use.iloc[0]), 1)
RESULTS["churn_highest_use_quintile"] = round(float(by_use.iloc[-1]), 1)
RESULTS["churn_status_active"] = round(float(by_status.loc[1]), 1)
RESULTS["churn_status_inactive"] = round(float(by_status.loc[2]), 1)
RESULTS["churners_who_complained"] = round(float(tr.loc[tr["Churn"] == 1, "Complains"].mean()) * 100, 1)
RESULTS["churners_inactive"] = round(float((tr.loc[tr["Churn"] == 1, "Status"] == 2).mean()) * 100, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.bar([fa("بدون شکایت"), fa("با شکایت")], by_complain.values, color=[GRAY, NAVY], width=0.55)
for i, v in enumerate(by_complain.values):
    ax.text(i, v + 1.5, fa(f"{v:.1f}٪"), ha="center", fontsize=11)
ax.set_title(fa("نرخ ریزش بر اساس شکایت"), fontsize=12)
ax.set_ylabel(fa("درصد مشتری‌هایی که رفتند"))
ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax = axes[1]
labels = [fa(s) for s in ["کم‌مصرف‌ترین", "۲", "۳", "۴", "پرمصرف‌ترین"]]
ax.bar(labels, by_use.values, color=[NAVY] + [GRAY] * 4, width=0.6)
for i, v in enumerate(by_use.values):
    ax.text(i, v + 1, fa(f"{v:.1f}٪"), ha="center", fontsize=10)
ax.set_title(fa("نرخ ریزش در پنج گروه مساوی مدت مکالمه"), fontsize=12)
ax.set_ylabel(fa("درصد مشتری‌هایی که رفتند"))
ax.set_ylim(0, max(by_use.values) * 1.25)
for a_ in axes:
    a_.set_yticks(a_.get_yticks()); a_.set_yticklabels([fa(f"{v:.0f}") for v in a_.get_yticks()])
fig.tight_layout()
save(fig, "churn-drivers", "نرخ ریزش مشتری‌های دارای شکایت در برابر بدون شکایت، و نرخ ریزش در پنج گروه مدت مکالمه")"""),
("md", "## ۳. مدل‌ها و مبنا"),
("code", """def rule(X):
    # the rule a manager could write without a model, ranked so the budget goes to the strongest signal first:
    # complained and low use > complained > low use (lowest-use quintile of the training data) > everyone else
    cut = X_train["Seconds of Use"].quantile(0.2)
    return 2.0 * (X["Complains"] == 1) + 1.0 * (X["Seconds of Use"] <= cut)

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_scores = {k: cross_val_score(m, X_train, y_train, cv=cv, scoring="average_precision") for k, m in models.items()}
RESULTS["cv_pr_auc"] = {k: [round(float(v.mean()), 3), round(float(v.std()), 3)] for k, v in cv_scores.items()}
scores = {}
for k, m in models.items():
    m.fit(X_train, y_train)
    scores[k] = m.predict_proba(X_test)[:, 1]
scores["rule"] = rule(X_test) + np.random.default_rng(SEED).random(len(X_test)) * 1e-6   # random order inside each tier
RESULTS["cv_pr_auc"]"""),
("code", """def top_k(score, frac=0.10):
    k = int(round(len(score) * frac))
    idx = np.argsort(-score)[:k]
    caught = int(y_test.values[idx].sum())
    return k, caught, caught / k, caught / y_test.sum()

base = float(y_test.mean())
table = []
for name, s in scores.items():
    k, caught, prec, rec = top_k(s)
    table.append({"model": name, "pr_auc": round(float(average_precision_score(y_test, s)), 3),
                  "roc_auc": round(float(roc_auc_score(y_test, s)), 3), "contacted": k, "caught": caught,
                  "precision_top10": round(prec * 100, 1), "recall_top10": round(rec * 100, 1), "lift": round(prec / base, 2)})
k = int(round(len(y_test) * 0.10))
table.append({"model": "random", "pr_auc": round(base, 3), "roc_auc": 0.5, "contacted": k,
              "caught": round(k * base, 1), "precision_top10": round(base * 100, 1),
              "recall_top10": 10.0, "lift": 1.0})
RESULTS["test_table"] = table
# the same two conditions without an order: any flagged customer, picked at random within the budget
flat = ((X_test["Complains"] == 1) | (X_test["Seconds of Use"] <= X_train["Seconds of Use"].quantile(0.2))).astype(float)
k_, c_, p_, r_ = top_k(flat + np.random.default_rng(SEED).random(len(X_test)) * 1e-6)
RESULTS["rule_unranked"] = {"flagged": int(flat.sum()), "caught": c_, "precision_top10": round(p_ * 100, 1)}
# how coverage grows if the retention budget grows: share of churners found at 10/15/20/30% contacted
RESULTS["recall_at"] = {name: {str(int(f * 100)): round(top_k(s, f)[3] * 100, 1) for f in (0.10, 0.15, 0.20, 0.30)}
                        for name, s in scores.items()}
pd.DataFrame(table)"""),
("code", """fig, ax = plt.subplots(figsize=(7.5, 4.6))
names = {"boosting": "گرادیان بوستینگ", "logistic": "رگرسیون لجستیک", "rule": "قاعده‌ی ساده (اول شکایت، بعد کم‌مصرف)"}
colors = {"boosting": NAVY, "logistic": CYAN, "rule": GRAY}
order = np.argsort(-scores["boosting"])
for name in ("boosting", "logistic", "rule"):
    o = np.argsort(-scores[name])
    captured = np.cumsum(y_test.values[o]) / y_test.sum() * 100
    share = np.arange(1, len(o) + 1) / len(o) * 100
    ax.plot(share, captured, color=colors[name], lw=2.4 if name == "boosting" else 1.8, label=fa(names[name]))
ax.plot([0, 100], [0, 100], color=GRAY, ls="--", lw=1.2, label=fa("انتخاب تصادفی"))
ax.axvline(10, color="#D1D5DB", lw=1)
ax.set_xlabel(fa("درصد مشتری‌هایی که با آن‌ها تماس گرفته می‌شود"))
ax.set_ylabel(fa("درصد مشتری‌های رونده که پیدا می‌شوند"))
ax.set_title(fa("با تماس با ۱۰٪ مشتری‌ها، چند درصد رونده‌ها پیدا می‌شوند؟"), fontsize=12)
ax.set_xticks([0, 10, 20, 30, 40, 60, 80, 100]); ax.set_xticklabels([fa(t) for t in [0, 10, 20, 30, 40, 60, 80, 100]])
ax.axvline(20, color="#E5E7EB", lw=1)
ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_yticklabels([fa(t) for t in [0, 20, 40, 60, 80, 100]])
ax.legend(frameon=False, fontsize=9, loc="lower right")
save(fig, "gain-curve", "منحنی پوشش: درصد مشتری‌های رونده که با تماس با درصد معینی از مشتری‌ها پیدا می‌شوند، برای مدل‌ها، قاعده‌ی ساده و انتخاب تصادفی")"""),
("md", "## ۴. کدام ستون بیشترین اثر را بر کیفیت پیش‌بینی دارد؟ (اهمیت جایگشتی روی داده‌ی آزمون؛ ستون‌های همبسته اهمیت را بین خود تقسیم می‌کنند)"),
("code", """pi = permutation_importance(models["boosting"], X_test, y_test, scoring="average_precision", n_repeats=20, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=features).sort_values()
RESULTS["top_features"] = [[k, round(float(v), 3)] for k, v in imp.sort_values(ascending=False).head(5).items()]
fa_names = {"Complains": "شکایت", "Seconds of Use": "مدت مکالمه", "Status": "وضعیت فعال/غیرفعال", "Frequency of use": "تعداد تماس",
            "Subscription Length": "مدت اشتراک", "Customer Value": "ارزش مشتری", "Distinct Called Numbers": "تعداد شماره‌های متفاوت",
            "Frequency of SMS": "تعداد پیامک", "Call Failure": "تماس ناموفق", "Charge Amount": "سطح شارژ", "Age": "سن", "Tariff Plan": "نوع تعرفه"}
top = imp.tail(8)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([fa(fa_names.get(i, i)) for i in top.index], top.values, color=[GRAY] * (len(top) - 4) + [NAVY] * 4)
ax.set_xlabel(fa("افت PR-AUC وقتی این ستون به‌هم ریخته شود"))
ax.set_title(fa("کدام ستون بیشترین اثر را بر کیفیت پیش‌بینی دارد؟"), fontsize=12)
xt = np.arange(0, top.max() + 0.05, 0.05)
ax.set_xticks(xt); ax.set_xticklabels([fa(f"{t:.2f}") for t in xt])
save(fig, "feature-importance", "اهمیت جایگشتی ستون‌ها در مدل گرادیان بوستینگ: افت PR-AUC وقتی هر ستون به‌هم ریخته شود")"""),
("md", """## ۵. آزمون پایداری: آیا نتیجه به یک ستون یا یک تقسیم خوش‌شانس وابسته است؟

دو بررسی: (الف) مدل بدون «وضعیت فعال/غیرفعال» و «ارزش مشتری»؛ مشتری‌ای که در ۹ ماه اول غیرفعال شده عملاً رفته، و ارزش مشتری از همان مصرف ساخته شده. (ب) تکرار ارزیابی گرادیان بوستینگ با ۲۰ تقسیم تصادفی متفاوت."""),
("code", """def evaluate(cols, seeds=range(20)):
    out = []
    for s in seeds:
        Xa, Xb, ya, yb = train_test_split(X[cols], y, test_size=0.25, stratify=y, random_state=s)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED).fit(Xa, ya)
        p = m.predict_proba(Xb)[:, 1]
        k = int(round(len(p) * 0.10)); idx = np.argsort(-p)[:k]
        out.append((average_precision_score(yb, p), yb.values[idx].mean() * 100, yb.values[idx].sum() / yb.sum() * 100))
    a = np.array(out)
    return {"pr_auc": [round(float(a[:, 0].mean()), 3), round(float(a[:, 0].std()), 3)],
            "precision_top10": [round(float(a[:, 1].mean()), 1), round(float(a[:, 1].std()), 1)],
            "recall_top10": [round(float(a[:, 2].mean()), 1), round(float(a[:, 2].std()), 1)]}

reduced = [c for c in features if c not in ("Status", "Customer Value")]
RESULTS["robust_all_features"] = evaluate(features)
RESULTS["robust_without_status_value"] = evaluate(reduced)
RESULTS["robust_all_features"], RESULTS["robust_without_status_value"]"""),
("md", """## ۶. اگر ردیف‌های تکراری حذف نمی‌شدند

همان ارزیابی ۲۰ تقسیمی، این بار روی داده‌ی خام با ۳۰۰ ردیف تکراری. می‌شماریم چند مشتری آزمون نسخه‌ی یکسانی در داده‌ی آموزش دارند و نمره چقدر عوض می‌شود."""),
("code", """Xr, yr = raw[features], raw["Churn"]
leak = []
for s in range(20):
    Xa, Xb, ya, yb = train_test_split(Xr, yr, test_size=0.25, stratify=yr, random_state=s)
    seen = set(map(tuple, Xa.assign(Churn=ya).values))
    twins = sum(tuple(r) in seen for r in Xb.assign(Churn=yb).values)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED).fit(Xa, ya)
    p = m.predict_proba(Xb)[:, 1]
    leak.append((twins, len(Xb), average_precision_score(yb, p)))
a = np.array(leak, dtype=float)
RESULTS["with_duplicates"] = {"test_rows_with_twin_in_train": [round(float(a[:, 0].mean()), 1), int(a[0, 1])],
                              "pr_auc": [round(float(a[:, 2].mean()), 3), round(float(a[:, 2].std()), 3)]}
RESULTS["with_duplicates"]"""),
("code", """with open("results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)
RESULTS"""),
]

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
nbf.write(nb, "analysis.ipynb")
print("analysis.ipynb:", len(nb.cells), "cells")
