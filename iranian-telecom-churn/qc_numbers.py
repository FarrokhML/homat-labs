"""QC: every number in ARTICLE.md must trace to results.json (or be a listed structural constant).

Run after run.py. Exits non-zero and lists any number it cannot trace.
"""
import json, re, sys
sys.stdout.reconfigure(encoding="utf-8")

FA = str.maketrans("۰۱۲۳۴۵۶۷۸۹٫٬", "0123456789.,")
r = json.load(open("results.json", encoding="utf-8"))

allowed = set()
def add(v):
    if isinstance(v, (int, float)):
        allowed.add(round(float(v), 3))
    elif isinstance(v, list):
        for x in v: add(x)
    elif isinstance(v, dict):
        for x in v.values(): add(x)
add(r)
t = {row["model"]: row for row in r["test_table"]}
derived = {
    round(r["churn_complained"] / r["churn_not_complained"]): "complaint ratio (~8x)",
    round(t["boosting"]["lift"], 1): "boosting lift 1dp", round(t["logistic"]["lift"], 1): "logistic lift 1dp",
    round(t["rule"]["lift"], 1): "rule lift 1dp", round(t["random"]["caught"]): "random caught",
    round(100 - r["churn_rate"]): "trivial accuracy", round(r["churn_rate"]): "churn rate rounded",
    round(r["test_churners"] / r["n_test"] * 100): "test churn share rounded",
}
for _, v in r["top_features"]:
    derived[round(v, 2)] = "feature importance 2dp"
w = r["with_duplicates"]["test_rows_with_twin_in_train"]
derived[round(w[0] / w[1] * 100)] = "share of test rows with a twin"
derived[t["boosting"]["caught"] - t["rule"]["caught"]] = "extra churners found by boosting vs rule"
derived[round(t["rule"]["lift"])] = "rule lift rounded (~5x)"
derived[round(r["churn_complained"])] = "complainer churn rounded"
for k in derived: allowed.add(float(k))
for name, d in r["recall_at"].items():
    for budget in d: allowed.add(float(budget))
# structural constants of the study design, stated in the notebook, not results
structural = {9: "months of behaviour", 12: "label month", 20: "seeds / top-20% budget", 10: "top-10% budget",
              5: "CV folds / quintiles", 3: "three approaches", 2: "two models", 1: "rank/one", 4: "four steps",
              563: "UCI dataset id", 6: "six-month example window", 15: "budget example", 0: "zero"}
for k in structural: allowed.add(float(k))

text = open("ARTICLE.md", encoding="utf-8").read().split("---", 2)[2]
text = re.sub(r"\(https?://[^)]*\)|DOI: ?[\d./A-Z]+|CC BY 4\.0|PR-AUC", " ", text)
bad = []
for m in re.finditer(r"[۰-۹][۰-۹٫٬]*", text):
    raw = m.group().rstrip("٫٬")
    v = float(raw.translate(FA).replace(",", ""))
    if round(v, 3) not in allowed:
        bad.append((m.group(), text[max(0, m.start() - 30): m.end() + 10].replace("\n", " ")))
print(f"checked {len(re.findall(r'[۰-۹][۰-۹٫٬]*', text))} numbers; untraced: {len(bad)}")
for b in bad: print("  ", b)
sys.exit(1 if bad else 0)
