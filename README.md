# Subway Incident Triage: predicting major delays at the moment an incident is logged

When something goes wrong on Toronto's subway, a control-centre operator has to decide quickly: is this a 3-minute blip, or the start of a 30-minute disruption that needs shuttle buses and supervisors? This project trains a model that estimates, at the moment an incident is logged, the probability that it becomes a **major delay (15+ minutes)**. It then turns that probability into an alert rule chosen by expected cost, not by accuracy.

**Live demo:** [reezwan95.github.io/#triage](https://reezwan95.github.io/#triage) lets you set an incident's code, station and time and get a prediction. The demo runs the exported logistic model directly in your browser.

## Results on unseen 2025 data

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Code base rate (lookup rule) | 0.267 | 0.777 | 0.0765 |
| Logistic regression (browser model) | 0.294 | 0.788 | 0.0764 |
| **LightGBM** | **0.301** | **0.792** | **0.0761** |

About 9.5% of delaying incidents in 2025 became major. A PR-AUC of 0.30 is therefore about **3.2× better than random**.

**Decision layer.** The costs are assumptions: a false alarm costs 1 unit and a missed major incident costs 8. Under those costs, the threshold picked on the 2024 validation set is **0.09**. On 2025 data that rule:

- catches **74%** of major incidents with about **8.7 alerts a day**, against 2.3 actual majors a day;
- cuts expected cost by **36.5%** compared with the better of "always alert" and "never alert";
- costs **10% less** than using the delay-code lookup alone.

**Calibration.** The predicted probabilities track observed rates closely. For example, incidents scored around 22% became major 21% of the time. The full calibration table and threshold sweep are in `reports/metrics.json`.

**What drives risk (mean |SHAP|).** The delay code dominates. Station, cause category and hour of day follow.

## Honest takeaways

- **The delay code carries most of the signal.** A well-built lookup table gets you most of the way. The model adds a modest but real gain from station and time context. Knowing when ML is worth it is part of the job.
- **The threshold matters more than the algorithm.** Moving from 0.09 to 0.30 cuts alerts from about 9 a day to about 2, but recall drops from 74% to 27%.
- **Leakage is avoided on purpose.** `min_gap` (the gap between trains) is measured after the fact, so it is excluded. The split is by time: train on Jan 2022 – Jun 2024, validate on Jul – Dec 2024, test on Jan – Oct 2025.

## Method

1. **Data.** Cleaned TTC Subway Delay Data (City of Toronto Open Data), Lines 1, 2 and 4, Jan 2022 – Oct 2025. There are 33,146 incidents that delayed a train. Cleaning is in [ttc-subway-delay-analysis](https://github.com/Reezwan95/ttc-subway-delay-analysis).
2. **Features known at logging time.** Delay code, cause category, station, line, direction, hour, day of week, month, a peak flag and a weekend flag. Rare codes and stations are collapsed into "Other" using training data only.
3. **Models.**
   - A code base-rate baseline.
   - L2-regularized logistic regression on one-hot features.
   - LightGBM with native categorical handling and early stopping on the validation set.
4. **Evaluation.** PR-AUC, which suits a 9.5% positive class, plus ROC-AUC, the Brier score, calibration, and a cost-based threshold chosen on validation and reported on test.
5. **Explainability.** SHAP values (TreeExplainer) on a sample of 4,000 test incidents.
6. **Deployment.** The logistic model's coefficients are exported to `reports/browser_model.json` (184 weights), so the portfolio demo needs no server.

## Run it

```bash
pip install pandas numpy scikit-learn lightgbm shap
# produce the cleaned CSV with the ttc-subway-delay-analysis pipeline, then:
python src/train.py ttc_subway_delays_clean_2022_2025.csv reports
```

The script writes `reports/metrics.json`, `reports/browser_model.json` and `reports/lightgbm_model.txt`. The last file is not committed because of its size; rerun the script to recreate it.

## Limitations

- The incident log has hour of day but no minute-level timestamps, so live system load can't be used as a feature yet.
- The 1:8 cost ratio is an illustrative assumption. A real deployment would set it with operations staff, and the sweep in `metrics.json` shows how results change with it.
- This is a portfolio project on public data. It is not a TTC system.

---
Reezwan-Us Sami · BASc Industrial Engineering (AI & Applied ML), University of Toronto · [Portfolio](https://reezwan95.github.io) · [LinkedIn](https://www.linkedin.com/in/reezwans)
