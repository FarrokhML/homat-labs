# Churn prediction on a real Iranian telecom dataset

Code behind the Homat article «کدام مشتری‌ها می‌روند؟» ([ARTICLE.md](ARTICLE.md)).

**Question:** with a retention budget for 10% of customers, which ones should we contact, and how much better is that than random choice or a one-line rule?

## Result (held-out test set, 713 customers, 112 churners, 71 contacted)

| Approach | Churners found in top 10% | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Random | ~11 | 15.7% | 10.0% | 0.157 |
| Rule, ranked: complained & low use > complained > low use | 54 | 76.1% | 48.2% | 0.712 |
| Logistic regression | 55 | 77.5% | 49.1% | 0.777 |
| Gradient boosting | 71 | 100% | 63.4% | 0.960 |

Over 20 random splits, boosting averages 98.0% ± 1.7 precision in the top 10%; without `Status` and `Customer Value`, 96.9% ± 1.9. The same rule without the ranking (random 71 of the 177 flagged) finds only 38. The dataset is unusually separable, so treat these numbers as an upper bound for real company data.

## Reproduce

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
python build_notebook.py   # writes analysis.ipynb from plain-Python cell definitions
python run.py              # executes the notebook; writes figures/ and results.json
python qc_numbers.py       # fails if any number in ARTICLE.md does not trace to results.json
```

## Data

*Iranian Churn Dataset*, UCI Machine Learning Repository, DOI [10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z), licensed CC BY 4.0. Included unmodified as `data/customer-churn.csv`. 300 exact duplicate rows are removed in the notebook before splitting.

Charts use [Vazirmatn](https://github.com/rastikerdar/vazirmatn) v33.003 (SIL Open Font License, `fonts/OFL.txt`).
