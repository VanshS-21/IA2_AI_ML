from pathlib import Path
import json
import pandas as pd
import numpy as np
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
T=OUT/'tables'
res=json.loads((OUT/'summary.json').read_text())
mart=pd.read_csv(T/'orders_mart.csv')
d=mart.loc[mart.delivered.eq(True)].copy().reset_index(drop=True)
rfm=pd.read_csv(T/'customer_rfm.csv')
d['segment']=d.customer_unique_id.map(rfm.set_index('customer_unique_id').segment)
d['cid']=pd.factorize(d.customer_unique_id)[0]
cols=['month','customer_state','gmv','freight','cid','review_score','late','segment']
rows=json.loads(d[cols].to_json(orient='values',double_precision=5))
ix=pd.Series(d.index,index=d.order_id)
items=pd.read_csv(T/'delivered_item_mart.csv')
items['idx']=items.order_id.map(ix)
ir=json.loads(items[['idx','category','seller_id','price']].to_json(orient='values',double_precision=2))
pay=pd.read_csv(ROOT/'data/raw/olist_order_payments_dataset.csv')
pay=pay.loc[pay.order_id.isin(ix.index)].copy();pay['idx']=pay.order_id.map(ix)
payments=json.loads(pay[['idx','payment_type','payment_value','payment_installments']].to_json(orient='values',double_precision=2))
pred=pd.read_csv(T/'test_predictions.csv');bins=np.linspace(0,1,31)
hist={'centers':((bins[:-1]+bins[1:])/2).tolist(),
      'ontime':np.histogram(pred.loc[pred.late.eq(0),'risk_score'],bins)[0].tolist(),
      'late':np.histogram(pred.loc[pred.late.eq(1),'risk_score'],bins)[0].tolist()}
def records(name):return json.loads(pd.read_csv(T/f'{name}.csv').to_json(orient='records'))
data={'orders':rows,'items':ir,'payments':payments,'results':res,'metrics':records('model_test'),
      'importance':records('feature_importance')[:10],'calibration':records('risk_calibration'),'hist':hist}
html=(ROOT/'src/dashboard_template.html').read_text(encoding='utf-8')
html=html.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(data,separators=(',',':'),allow_nan=False))
(ROOT/'dashboard').mkdir(exist_ok=True)
(ROOT/'dashboard/index.html').write_text(html,encoding='utf-8')
print('Dashboard written:',len(html),'characters')
