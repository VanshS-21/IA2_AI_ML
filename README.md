# IA2: Olist Marketplace Analytics and Late-Delivery Prediction

An end-to-end data analytics and data science assessment addressing:

> How can Olist increase marketplace growth while improving customer satisfaction and reducing delivery-related problems?

The submission combines marketplace analysis, customer segmentation, statistical testing, delivery-risk classification and three quantified business recommendations. Findings describe the supplied **2016–2018 historical dataset**. Monetary values are in **Brazilian reais (BRL)**.

## Submission deliverables

| Deliverable | File | Contents |
| --- | --- | --- |
| Data analysis | [Executed notebook](notebooks/olist_analysis.ipynb) | Data understanding, cleaning, EDA, customer/revenue analytics, statistical tests, feature engineering, model training, evaluation and interpretation |
| SQL | [Analysis queries](results/analysis.sql) | Order-level reconciliation and seller aggregations |
| Dashboard | [Download interactive dashboard](https://github.com/VanshS-21/IA2_AI_ML/raw/refs/heads/main/dashboard/index.html) | Four management pages with state/month filters, metric definitions and KPI export |
| Presentation | [Leadership presentation — 10 slides](reports/presentation.pptx) | Business problem, methodology, findings, statistics, ML performance and recommendations; editable charts and tables |
| Executive summary | [One-page summary](reports/executive_summary.pdf) | What leadership should do next, evidence, ownership and opportunity scenarios |
| Supporting results | [Result tables](results/tables), [full-precision summary](results/summary.json) | Revenue, customers, reviews, delivery performance, model metrics and interpretation |
| Trained model | [Selected pipeline](results/models/late_delivery.joblib) | Logistic regression pipeline, feature list and frozen alert threshold |

**Review order:** executive summary → presentation → dashboard → notebook. Download the HTML dashboard and open it in a browser; GitHub's file viewer does not execute HTML. The dashboard embeds its data and chart library and works offline. The presentation is also available in [Canva](https://canva.link/hww5iuhf1rew8s5); the submitted PowerPoint is included in this repository.

## Assessment coverage

| Assessment area | Evidence |
| --- | --- |
| **Objective A — Understand the business** | Revenue/order trends, AOV, categories, states, sellers, payments, customers, review distribution and delivery performance in notebook sections 1–4 and the dashboard |
| **Objective B — Customer and revenue analytics** | Unique customers, frequency, repeat purchases, RFM-style segmentation, mature-cohort retention, freight, payment methods and installments in sections 4–5 |
| **Objective C — Satisfaction and delivery** | Late rate, on-time/late review comparison, delay correlations and category/state/seller comparisons in section 6 |
| **Statistical analysis** | Welch's t-test and Spearman correlation, with hypotheses, significance level, statistics, p-values and interpretation; robustness checks and confidence intervals |
| **Objective D — Machine learning** | Leakage controls, feature engineering, chronological splits, logistic regression, random forest and a baseline in sections 7–8 |
| **Evaluation and interpretation** | Accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrices, coefficients and permutation importance |
| **Visualization** | Executive overview, revenue/customers, customer experience and machine-learning dashboard pages; notebook figures |
| **Business recommendations** | Three prioritized changes with owners, timing, measurable outcomes and scenario assumptions in sections 9–11, the summary and presentation |

## Five key business findings

| Finding | Evidence | Business implication |
| --- | --- | --- |
| Growth mainly follows order volume | Jan–Jul 2018 GMV grew **161.6%** and delivered orders grew **160.8%** versus Jan–Jul 2017 | Separate customer/order acquisition from basket-value improvement |
| Commercial value is concentrated | Top five categories contribute **39.8%** of GMV; SP, RJ and MG contribute **63.4%**; top ten sellers contribute **13.3%** | Prioritize improvements where commercial exposure is largest |
| Repeat purchasing is limited | **3.00%** of delivered customers repeat; mature-cohort 90-day repeat is **2.28%** | Test second-purchase interventions with equal follow-up and a holdout |
| Late deliveries are associated with poor experience | **8.11%** of eligible deliveries are late; mean review is **2.57** for late orders versus **4.29** on time | Investigate seller/route reliability and validate interventions experimentally |
| Checkout features provide modest early-warning value | Selected model July ROC-AUC **0.690**, average precision **0.097**, recall **23.6%**, precision **10.3%** | Run a capacity-limited pilot and assess intervention costs before scaling |

### Marketplace metrics

| Metric | Result |
| --- | ---: |
| All orders | 99,441 |
| Delivered orders | 96,478 |
| Unique customers, all statuses | 96,096 |
| Unique delivered customers | 93,358 |
| Delivered merchandise GMV | BRL 13,221,498.11 |
| Average delivered order value | BRL 137.04 |
| Delivered freight value | BRL 2,198,275.64 |
| Average latest review | 4.16 / 5 |
| Late / eligible deliveries | 7,826 / 96,470 |
| Repeat delivered customers | 2,801 |
| Credit-card share of payment value | 78.5% |

Leading categories are health and beauty, watches and gifts, bed/bath/table, sports and leisure, and computer accessories. Rio de Janeiro combines substantial value with a **13.5%** late rate, versus **5.9%** in São Paulo and **5.6%** in Minas Gerais. Complete rankings appear in [categories](results/tables/categories.csv), [states](results/tables/states.csv) and [sellers](results/tables/sellers.csv).

## Data and preparation

The nine source CSVs are from the [assessment dataset folder](https://drive.google.com/drive/folders/1qmq0FkVqotNkjWcqJP1Wg_rntzlSvfs5). Compressed copies are included in [data/source](data/source), so the repository contains the data needed for reproduction. The [source manifest](data/source_manifest.json) records filenames, Drive IDs, original sizes and SHA-256 hashes. Uncompressed CSVs and large generated marts are excluded from Git and restored or rebuilt with the commands below.

| Source | Grain / role |
| --- | --- |
| Orders | One record per order; timestamps and status |
| Order items | One record per order item; price, freight, product and seller |
| Payments | One record per payment sequence; method, value and installments |
| Reviews | Review records; an order may have multiple reviews |
| Customers | Order-linked IDs, persistent customer identity and location |
| Sellers | Seller identity and location |
| Products | Category, weight and dimensions |
| Category translation | Portuguese-to-English category labels |
| Geolocation | Multiple coordinate records per ZIP prefix |

Items and payments are aggregated independently before joining to orders. Customer behaviour uses `customer_unique_id`; the latest answered review represents each reviewed order. ZIP prefixes retain leading zeros. Geolocation becomes one median coordinate pair per ZIP after duplicate and validity checks. Missingness, orphan keys, timestamps and payment reconciliation are checked explicitly.

Quality findings include **547** orders with multiple reviews, **8** delivered orders without actual delivery timestamps, **261,831** exact geolocation duplicates and **303** payment-versus-item/freight differences above one cent. Differences are reported without artificial balancing adjustments.

### Metric definitions

- **Revenue / GMV:** delivered item prices excluding freight; merchandise value, not platform net revenue or profit.
- **AOV:** delivered merchandise GMV divided by delivered orders.
- **Late delivery:** actual delivery timestamp exceeds estimated delivery timestamp; denominator is delivered orders with valid timestamps. A calendar-day sensitivity analysis is included.
- **Review mean:** latest answered review per order, excluding missing reviews.
- **Repeat rate:** customers with at least two delivered orders divided by unique delivered customers in the stated scope.
- **90-day repeat:** a second purchase within 90 days among customers with at least 90 days of observation.
- **RFM segmentation:** snapshot on 30 August 2018; recent means within 90 days, and high value means customer GMV at or above the 75th percentile, BRL 154.74.

## Statistical evidence

Both primary analyses use **α = 0.05** and eligible delivered orders with reviews.

| Analysis | Hypotheses | Result | Interpretation |
| --- | --- | --- | --- |
| Welch's two-sample t-test | H₀: on-time and late orders have equal mean reviews. H₁: their means differ. | t = **−89.41**, df = **8,309.03**, p **< 10⁻³⁰⁰**. Mean gap **−1.73** points; 95% CI **[−1.77, −1.69]**. Hedges' g **−1.45**. | Reject H₀. The gap is large in this extract, although reviews are ordinal and the data are observational. |
| Spearman rank correlation | H₀: signed delay and review score have zero rank correlation. H₁: the correlation differs from zero. | ρ = **−0.176**, n = **95,824**, p **< 10⁻³⁰⁰**. | Reject H₀. Longer signed delays have a modest negative monotonic association with reviews; significance alone does not establish a strong predictor. |

Mann–Whitney, alternative review selection, calendar-day lateness and state/month stratification provide robustness checks. Product mix, sellers, geography and unmeasured factors may influence both delivery and reviews; the associations do **not** establish that lateness alone causes the review gap.

## Late-delivery prediction

**Prediction moment:** checkout complete. **Target:** `Late_Delivery = 1` when actual delivery is after estimated delivery; otherwise 0 for eligible delivered orders.

### Features and leakage prevention

Features include order value, freight/ratio, item/seller/category counts, average item price, weight, volume, approximate distance, cross-state indicator, promised delivery days, payment method/installments, purchase hour/weekday, cyclical month, locations, category and strictly prior customer order count/spend.

Actual delivery, carrier handoff, approval timestamps, reviews, comments/dates, observed delay and future outcomes are excluded. Customer history collapses equal purchase timestamps so simultaneous orders cannot reveal one another. Median imputation, scaling and rare-category encoding fit on training data only. Catalog, seller-assignment, payment and delivery-promise availability at checkout remain assumptions requiring event-log validation.

| Split | Purchase window | Eligible labelled orders | Role |
| --- | --- | ---: | --- |
| Training | Before January 2018 | 41,197 | Fit using labels observed before 1 January 2018 |
| Validation | January–March 2018 | 16,782 | Choose model/threshold using labels observed before 1 April 2018 |
| Diagnostic | April–June 2018 | 19,643 | Diagnose calendar instability; informed feature correction before July evaluation |
| Final test | July 2018 | 6,156 | Evaluate frozen models and thresholds |

Logistic regression and random forest are compared with an always-on-time baseline. **Validation average precision selects the model.** Validation F2 selects its threshold under an assumed maximum **20% alert capacity**. Recall receives extra weight because missed late orders prevent proactive intervention; the capacity limit constrains false-alert workload. Actual costs are unavailable, so this is a pilot policy rather than an economically optimal threshold.

### Final July test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision | Alert rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Logistic regression — selected** | 87.3% | 10.3% | 23.6% | 0.143 | 0.690 | 0.097 | 10.3% |
| Random forest | 77.0% | 9.7% | 49.6% | 0.162 | 0.691 | 0.102 | 23.0% |
| Always on time | 95.5% | 0.0% | 0.0% | 0.000 | 0.500 | 0.045 | 0.0% |

The baseline's high accuracy hides zero late-order detection. Logistic regression remains selected because it won validation; the final test is not used to reselect the model. Its average precision is **2.17×** July late prevalence. The frozen threshold is **0.683824**.

| Actual outcome | No alert | Alert |
| --- | ---: | ---: |
| On time | 5,311 | 569 |
| Late | 211 | 65 |

Permutation importance identifies promised delivery days, customer state, average item price, cross-state routing and distance as leading validation signals. These support route/promise diagnostics, not causal attribution. See [importance](results/tables/feature_importance.csv), [coefficients](results/tables/logistic_coefficients.csv), [validation metrics](results/tables/model_validation.csv) and [test metrics](results/tables/model_test.csv). Class-weighted scores require calibration before interpretation as probabilities.

## Three changes leadership should prioritize

| Priority | Action, owner and timing | Evidence and illustrative opportunity | Success measure |
| --- | --- | --- | --- |
| **1. Controlled delivery-risk pilot** | Operations, first 30 days: intervene on high-risk alerts with a randomized holdout and daily capacity limit. | The threshold flags **634** July orders, including **65** actually late orders. Preventing 20% of those outcomes would avoid **13** late deliveries in that cohort. | Late outcomes prevented per intervention, reviews, alerts/day and cost per avoided late delivery |
| **2. Seller and route diagnostics** | Logistics, days 30–60: review **35** sellers with at least 100 eligible orders and a Wilson lower bound above the overall late rate; validate current root causes. | The deduplicated portfolio contains **1,781** late orders. A 20% reduction implies about **356** fewer late outcomes and **BRL 44,192** of associated merchandise exposure. | Late rate and review improvement versus a comparison group, by seller and route |
| **3. Second-purchase experiment** | CRM, days 60–90: test follow-up for **4,226** recent high-value first-time buyers. | A one-percentage-point incremental repeat uplift implies about **42** additional orders and **BRL 5,791** GMV at overall AOV, before incentives. | Incremental 90-day repeat purchases and contribution after offer costs |

These are assumption-based scenarios, not forecasts. Merchandise exposure is not recoverable revenue. Portfolios may overlap, so scenario values must not be added.

## Repository structure

```text
IA2_AI_ML/
├── README.md
├── requirements.txt              # Direct dependency versions
├── requirements.lock.txt         # Full resolved environment
├── data/
│   ├── source_manifest.json      # Source IDs, sizes and hashes
│   └── source/*.csv.gz           # Nine compressed original CSVs
├── notebooks/olist_analysis.ipynb # Executed analysis
├── dashboard/index.html          # Self-contained dashboard
├── reports/
│   ├── executive_summary.pdf     # One page
│   ├── presentation.pptx         # Ten slides
│   └── presentation_notes.md     # Detailed slide narrative
├── results/
│   ├── analysis.sql
│   ├── summary.json
│   ├── models/late_delivery.joblib
│   └── tables/                   # Supporting evidence
└── src/
    ├── analysis.py               # Cell-marked source
    ├── build_notebook.py         # Build/execute notebook
    ├── build_dashboard.py
    ├── dashboard_template.html
    ├── build_summary.py
    ├── download_data.py          # Fetch/verify source CSVs
    └── verify_submission.py      # Validate deliverables
```

## Reproduce the analysis

The notebook contains executed outputs. Reproduction uses **Python 3.14** and the versions in `requirements.txt`.

```bash
git clone https://github.com/VanshS-21/IA2_AI_ML.git
cd IA2_AI_ML
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies, register the notebook kernel and extract/verify source data:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name olist-assessment --display-name "Python (Olist Assessment)"
python src/download_data.py
```

Rebuild the notebook, results, dashboard and summary, then validate:

```bash
python src/build_notebook.py
python src/build_dashboard.py
python src/build_summary.py
python src/verify_submission.py
```

Alternatively, run `python -m jupyterlab notebooks/olist_analysis.ipynb` and execute cells in order. Run commands from the repository root. Installing dependencies initially requires internet; source extraction, analysis and the dashboard then work locally. If a bundled source archive is absent, the data script attempts to restore that CSV from Drive. The PowerPoint is supplied as the final editable presentation.

Raw CSVs are extracted to `data/raw/`. The analysis regenerates `orders_mart.csv`, `delivered_item_mart.csv` and `customer_rfm.csv` in `results/tables/`; these large intermediate tables are intentionally ignored by Git.

## Limitations

- Historical observational data do not establish causal effects or current marketplace performance.
- Delivered-only modelling excludes unresolved, cancelled and unavailable orders. The final test covers one month; April–June informed a feature correction.
- Partial windows and unequal follow-up affect growth/retention; the analysis includes matched growth windows and mature retention cohorts.
- Seller/geographic comparisons reflect mix and sample size. Confidence bounds support screening rather than claims of individual seller significance.
- Margin, platform take rate, refunds and intervention costs are unavailable. GMV cannot be interpreted as net revenue or profit.
- Checkout feature availability, calibration, temporal stability and capacity require current-data validation before rollout.
