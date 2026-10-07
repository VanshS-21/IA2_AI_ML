## 1. Olist marketplace decisions

Business question: how can Olist grow while improving satisfaction and reducing delivery failures? The three budget priorities are a controlled risk-alert pilot, seller/route diagnostics, and a second-purchase experiment.

## 2. Data and decision rules

Nine CSVs: orders, items, payments, reviews, customers, sellers, products, category translation and geolocation. 99,441 orders. Integrity assertions check keys, foreign keys, monetary reconciliation and temporal splits. Eight delivered orders lack actual delivery timestamps. 547 orders have multiple review rows. 261,831 exact geolocation duplicates are removed. 303 orders differ by more than one cent between recorded payments and item-plus-freight totals. No fabricated balancing adjustments.

## 3. Growth follows a much larger order base

Jan-Jul 2018 versus Jan-Jul 2017: GMV 6379548.48 versus 2438756.43 BRL. Orders 46432 versus 17805. Monthly GMV is attributed to purchase month. August 2018 is an incomplete observation window. Delivered-only recognition near the boundary can also censor recent orders.

## 4. Value concentrates in a few categories and states

Category revenue sums actual item prices on delivered orders. The top five categories account for 39.8% of GMV. SP, RJ and MG account for 63.4%. State uses customer location. Full seller ranking: top ten sellers account for 13.3% of delivered GMV.

## 5. Second purchases are a measurable growth test

Unique delivered customers: 93358. All-status unique customers: 96096. Repeat within the delivered extract: 2801 customers, 3.00%. Among 75320 customers with at least 90 days observation, 2.28% made another purchase within 90 days. RFM snapshot 2018-08-30, recent within 90 days, high value at or above customer GMV 75th percentile BRL 154.7375. Credit cards are 78.5% of payment value and 66.8% of credit-card rows use more than one installment.

## 6. Late orders receive much poorer reviews

Review sample: 88163 on-time orders, 7661 late orders. Poor scores 1-2: 9.2% on time, 54.1% late. Timestamp late rate 8.11%; calendar-day alternative 6.77%. State late rates: RJ 13.47%, SP 5.89%, MG 5.61% (eligible delivered orders). Observational relationship may reflect routes, sellers, seasonal demand and review selection.

## 7. Statistical evidence supports an intervention test

Alpha=0.05. Welch H0: mean review score equal between late and on-time groups. H1: means differ. t=-89.4121464612284, df=8309.03228521667, p=<1e-300. Late-minus-on-time gap -1.729044476766842, 95% CI [-1.7669516153158529,-1.6911373382178312], Hedges g=-1.4454613116962918. Spearman H0: rank correlation zero, H1: nonzero. n=95824, rho=-0.17593705467269516, p=<1e-300. Mann-Whitney robustness p=<1e-300. Mean-all-reviews gap -1.7277863340342652. Calendar-day definition review gap -2.0198955371964127. State/month adjusted descriptive odds ratio 10.712452639431298. Independence assumptions and ordinal review scores limit interpretation.

## 8. Prediction uses information available at checkout

Features: gmv, freight, freight_ratio, n_items, n_sellers, n_categories, avg_item_price, weight_g, volume_cm3, distance_km, cross_state, promise_days, installments, payment_rows, previous_orders, previous_spend, month_sin, month_cos, purchase_weekday, purchase_hour, customer_state, seller_state, primary_category, payment_type. Checkout feature availability is assumed and needs event-log validation, especially payment, catalog and seller assignment snapshots. Exclude actual delivery, carrier handoff, approval times, reviews, order status, realized delay and full-period seller outcome rates. Strict previous customer purchases collapse equal timestamps before cumulative counts/spend. Median imputation, scaling and infrequent category encoding fit only on train. Train: 41197 orders, 5.2% late; Validation: 16782 orders, 8.4% late; Diagnostic: 19643 orders, 5.1% late; Test: 6156 orders, 4.5% late. Model selection is validation AP, threshold selection validation F2 with at most 20% alerts. The initial Apr-Jun diagnostic revealed continuous-time/month collinearity. Replace it with cyclical month before inspecting July.

## 9. Prediction helps rank risk, with substantial uncertainty

Logistic won validation AP 0.1545895799. Selected threshold 0.6838238487759531. Test July 6156 orders, prevalence 0.044834308. ROC-AUC CI 0.6579454779711476,0.7200271190643038; AP CI 0.07906351195554581,0.1318559293234935. Selected confusion matrix TN 5311, FP 569, FN 211, TP 65. Brier 0.2232998840595819. Raw scores require calibration. RF achieves higher July recall, but was not selected because its validation AP was lower. Do not change selection after observing July. Top validation permutation features: promise_days, customer_state, avg_item_price, cross_state, distance_km. False alerts dominate at these thresholds, favoring low-cost interventions.

## 10. Three changes for the next operating budget

Recommendations are conditional experiments. Risk alert owner operations, 0-30 days. At July frozen threshold 634 alerts include 65 late orders. 20% intervention-success scenario means 13 fewer late outcomes. Seller route diagnostics owner logistics, days 30-60. 35 sellers meet >=100 eligible orders and Wilson lower bound above baseline. Deduplicated portfolio 14423 orders, 1781 late. 20% reduction scenario 356.20000000000005 fewer late outcomes and BRL 44191.874 exposure addressed, not recovered revenue. CRM owner days 60-90. 4226 recent high-value first buyers. One point incremental repeat lift implies 42.26 orders, BRL 5791.377413800037 GMV at market AOV before costs. These portfolios overlap. Do not add scenarios. No margin/take-rate/cost data.
