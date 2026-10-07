from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'
r=json.loads((OUT/'summary.json').read_text());s=r['summary'];t=r['statistics'];m=r['selected_model'];o=r['opportunity']
(ROOT/'reports').mkdir(exist_ok=True)
path=ROOT/'reports/executive_summary.pdf'
c=canvas.Canvas(str(path),pagesize=(595.28,841.89))
c.setTitle('Olist | Leadership decisions'); c.setAuthor('Olist Assessment Analysis')
ink=HexColor('#152b45');blue=HexColor('#245ecb');muted=HexColor('#54667a')
styles={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=ink),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=blue),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.8,leading=14,textColor=ink),
 'foot':ParagraphStyle('foot',fontName='Helvetica',fontSize=8,leading=11,textColor=muted)}
y=797; left=43;width=509
def p(text,style='body',gap=10):
    global y
    para=Paragraph(text,styles[style]);w,h=para.wrap(width,800);y-=h;para.drawOn(c,left,y);y-=gap

p('Olist leadership decisions','title',5)
p('Historical marketplace assessment | 2016-2018 | All amounts BRL','foot',17)
p('The business is growing. Delivery reliability and repeat purchases deserve the next operating budget.','head',10)
p(f'Delivered merchandise value totals <b>R$ {s["gmv"]/1e6:.2f}m</b> across <b>{s["delivered_orders"]:,} orders</b>, with AOV of <b>R$ {s["aov"]:.0f}</b>. Jan-Jul 2018 GMV grew {s["jan_jul_gmv_growth"]:.1%} versus Jan-Jul 2017, alongside {s["jan_jul_order_growth"]:.1%} order growth. The top five categories generate {s["top5_category_share"]:.1%} of GMV. SP, RJ and MG generate {s["top3_state_share"]:.1%}.')
p(f'<b>{s["late_rate"]:.2%}</b> of {s["delivery_eligible_n"]:,} eligible deliveries are late. Late orders average <b>{t["late_review_mean"]:.2f} stars</b> versus <b>{t["ontime_review_mean"]:.2f}</b> on time. The gap is {t["difference"]:.2f} points (95% CI {t["difference_ci"][0]:.2f} to {t["difference_ci"][1]:.2f}). Welch and Spearman tests both have p &lt; 0.001. This strong association supports an intervention experiment, not a causal revenue claim.',gap=17)
p('Three changes to fund first','head',10)
p('<b>1. Pilot early delivery-risk alerts. Owner: operations. First 30 days.</b><br/>'
  f'{m["model"]} won validation. On the July test it catches {m["recall"]:.1%} of late orders at {m["precision"]:.1%} precision, with ROC-AUC {m["roc_auc"]:.3f} and average precision {m["average_precision"]:.3f}. It flags {m["alert_rate"]:.1%} of orders. Start with a low-cost randomized intervention and a daily workload cap. {o["model_flagged_late_test_orders"]:,} flagged July orders were actually late. A 20% intervention-success scenario means about {o["model_scenario_20pct_flagged_late_avoided"]:.0f} fewer late outcomes. Track late orders avoided per intervention and customer reviews.')
p('<b>2. Investigate the highest-exposure seller and route failures. Owner: logistics. Days 30-60.</b><br/>'
  f'{o["priority_sellers"]} sellers have at least 100 eligible orders and a Wilson lower bound above marketplace lateness. Their deduplicated portfolio includes {o["priority_late_orders"]:,} late orders. A 20% reduction scenario corresponds to <b>{o["scenario_late_orders_avoided"]:.0f} fewer late outcomes</b> and <b>R$ {o["scenario_gmv_exposure_addressed"]:,.0f}</b> of associated merchandise exposure. Validate current-period persistence, seller handling and route capacity before action. Screening many sellers does not establish individual significance.')
p('<b>3. Test second-purchase offers. Owner: CRM. Days 60-90.</b><br/>'
  f'Only {s["repeat_rate"]:.1%} of delivered customers repeat within the extract. The mature-cohort 90-day repeat measure is {s["repeat_within_90days"]:.2%}. Target the {o["retention_target_customers"]:,} recent high-value first-time buyers identified at the snapshot. A one-percentage-point incremental repeat uplift implies about {o["retention_scenario_extra_orders"]:.0f} additional orders, or R$ {o["retention_scenario_extra_gmv_at_market_aov"]:,.0f} at overall AOV before offer costs. Use a holdout and measure incremental contribution after incentives.',gap=15)
p('Decision guardrails','head',7)
p('GMV is merchandise value, not platform net revenue or profit. No margin, refunds, take rate or intervention costs are available. Scenario values are illustrative and overlap, so do not add them. Historical delivered-only labels omit unresolved and cancelled orders. Scores need calibration and checkout feature availability needs event-log verification. April-June diagnostics informed a calendar-feature correction; July remained the final evaluation window. Revalidate with current data before rollout.','foot',9)
p('Source: the nine supplied Olist CSVs. Reproducible calculations, eligibility rules, statistical tests, model splits and uncertainty appear in the companion notebook and dashboard.','foot',0)
assert y>38,f'Executive summary overflows: y={y}'
c.setStrokeColor(HexColor('#dce5ef'));c.line(43,34,552,34);c.setFont('Helvetica',8);c.setFillColor(muted);c.drawString(43,22,'Olist assessment');c.drawRightString(552,22,'1 / 1')
c.showPage();c.save();print(path,'bottom y:',round(y,1))
