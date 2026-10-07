from pathlib import Path
import re
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'src/analysis.py').read_text(encoding='utf-8')
cells=[]
for chunk in re.split(r'^# %%',source,flags=re.MULTILINE):
    if not chunk.strip():continue
    if chunk.startswith(' [markdown]'):
        text='\n'.join(re.sub(r'^# ?', '', line) for line in chunk.splitlines()[1:]).strip()
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:cells.append(nbformat.v4.new_code_cell(chunk.strip()))
cells.append(nbformat.v4.new_markdown_cell('''## 10. Management visuals

The self-contained HTML dashboard provides four filterable management pages.
These notebook figures preserve the core evidence alongside the calculations.
'''))
cells.append(nbformat.v4.new_code_cell('''
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
ax[0,0].plot(monthly.month,monthly.gmv/1e6,color='#245ecb',marker='o',markersize=3)
ax[0,0].set(title='Delivered merchandise value',ylabel='BRL millions')
ax[0,0].tick_params(axis='x',rotation=60,labelsize=7)
z=categories.head(8).iloc[::-1]
ax[0,1].barh(z.category.str.replace('_',' '),z.gmv/1e6,color='#007d7c')
ax[0,1].set(title='Leading categories',xlabel='BRL millions')
ax[1,0].bar(['On time','Late'],[statistics['ontime_review_mean'],statistics['late_review_mean']],color=['#245ecb','#bd3d4b'])
ax[1,0].set(title='Mean latest review by delivery outcome',ylabel='Review score',ylim=(0,5))
z=states.loc[states.eligible.ge(100)].sort_values('late_rate').tail(8)
ax[1,1].barh(z.customer_state,z.late_rate*100,color='#bd3d4b')
ax[1,1].set(title='States with elevated delivery risk',xlabel='Late delivered orders (%)')
plt.show()
fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
cm=np.array([[best['tn'],best['fp']],[best['fn'],best['tp']]])
ax[0].imshow(cm,cmap='Blues')
for row in range(2):
    for col in range(2):ax[0].text(col,row,f'{cm[row,col]:,}',ha='center',va='center',color='white' if cm[row,col]>cm.max()/2 else '#152b45',fontsize=18)
ax[0].set(xticks=[0,1],xticklabels=['No alert','Alert'],yticks=[0,1],yticklabels=['On time','Late'],title='July 2018 confusion matrix')
z=imp.head(8).iloc[::-1]
ax[1].barh(z.feature,z.importance,xerr=z['std'],color='#007d7c')
ax[1].set(title='Validation permutation importance',xlabel='Average precision decrease')
plt.show()
'''))
cells.append(nbformat.v4.new_markdown_cell('## 11. Executive interpretation and next actions'))
cells.append(nbformat.v4.new_code_cell('''
from IPython.display import Markdown, display
b=result['selected_model']; op=result['opportunity']; ss=result['summary']; st=result['statistics']
text=f"""
Five findings matter most:

1. Delivered GMV is **BRL {ss['gmv']:,.0f}** across **{ss['delivered_orders']:,} orders**. Jan-Jul GMV grew **{ss['jan_jul_gmv_growth']:.1%}** versus the prior-year window, largely alongside order-volume growth (**{ss['jan_jul_order_growth']:.1%}**).
2. Five categories contribute **{ss['top5_category_share']:.1%}** of GMV. SP, RJ and MG contribute **{ss['top3_state_share']:.1%}**. Concentration identifies where improvements affect the most value.
3. **{ss['repeat_rate']:.1%}** of delivered customers repeat within the extract. The mature-cohort 90-day repeat measure is **{ss['repeat_within_90days']:.2%}**, so retention should be tested with equal follow-up windows.
4. **{ss['late_rate']:.2%}** of eligible deliveries are late. Their mean review is **{st['late_review_mean']:.2f}**, compared with **{st['ontime_review_mean']:.2f}** on time. The mean gap is **{st['difference']:.2f}** points (95% CI {st['difference_ci'][0]:.2f} to {st['difference_ci'][1]:.2f}). Observational data cannot establish causality.
5. **{b['model']}** won validation. July test ROC-AUC is **{b['roc_auc']:.3f}**, average precision **{b['average_precision']:.3f}**, recall **{b['recall']:.1%}**, and precision **{b['precision']:.1%}**. It flags **{b['alert_rate']:.1%}** of orders. This supports a controlled pilot, subject to workload and calibration checks.

**Three funded changes:**

- **Risk-alert pilot, operations owner, first 30 days.** Intervene on selected high-risk orders with a randomized holdout. Monitor late rate, review score, alerts/day and cost per late order avoided. At the frozen threshold, {op['model_flagged_test_orders']:,} July orders would receive alerts, including {op['model_flagged_late_test_orders']:,} actually late orders. A 20% intervention success assumption implies {op['model_scenario_20pct_flagged_late_avoided']:.0f} avoided late outcomes in that historical cohort. This scenario is not a forecast.
- **Seller/route review, logistics owner, 30-60 days.** Review {op['priority_sellers']} sellers meeting the volume and Wilson-bound screen. Their deduplicated portfolio includes {op['priority_late_orders']:,} late orders. A 20% reduction corresponds to about {op['scenario_late_orders_avoided']:.0f} fewer late outcomes and BRL {op['scenario_gmv_exposure_addressed']:,.0f} of associated merchandise exposure. Validate seller root causes and current-period stability before action.
- **Second-purchase experiment, CRM owner, 60-90 days.** Target {op['retention_target_customers']:,} recent high-value first-time buyers in the snapshot. A one-percentage-point incremental repeat uplift gives about {op['retention_scenario_extra_orders']:.0f} additional orders, or BRL {op['retention_scenario_extra_gmv_at_market_aov']:,.0f} at overall AOV before offer costs. Measure incremental margin, not just redemptions.

**Limits:** GMV is not platform revenue. No margin, take rate, refunds or intervention costs are available. The final model test is July 2018 and must be revalidated in current operations. April-June was diagnostic and informed the calendar correction. Delivered-only labels omit unresolved and cancelled orders. Verify checkout feature snapshots, calibrate scores, and run a randomized pilot before scaling.
"""
display(Markdown(text))
'''))
nb=nbformat.v4.new_notebook(cells=cells,metadata={
 'kernelspec':{'name':'olist-assessment','display_name':'Python (Olist Assessment)','language':'python'},
 'language_info':{'name':'python','version':'3.14.8'}})
path=ROOT/'notebooks/olist_analysis.ipynb'
path.parent.mkdir(exist_ok=True)
nbformat.write(nb,path)
client=NotebookClient(nb,timeout=1200,kernel_name='olist-assessment',resources={'metadata':{'path':str(ROOT)}})
client.execute()
nbformat.write(nb,path)
assert not any(o.get('output_type')=='error' for c in nb.cells for o in c.get('outputs',[]))
print(f'Executed {sum(c.cell_type=="code" for c in nb.cells)} code cells, saved {path}')
