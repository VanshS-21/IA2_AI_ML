"""Verify source integrity, result reconciliation and submission structure."""
from pathlib import Path
import hashlib
import json
import math
import re
import zipfile
import nbformat
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
checks = []
def check(condition, description):
    assert condition, description
    checks.append(description)

manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
for entry in manifest:
    path = ROOT / 'data/raw' / entry['name']
    check(path.stat().st_size == entry['bytes'] and
          hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'],
          f"Original CSV size and SHA-256 match: {entry['name']}")

r = json.loads((OUT / 'summary.json').read_text())
s = r['summary']
orders = pd.read_csv(OUT / 'tables/orders_mart.csv')
delivered = orders.loc[orders.delivered.eq(True)]
check(len(orders) == s['all_orders'] and len(delivered) == s['delivered_orders'],
      'Order mart reconciles to 99,441 total and 96,478 delivered orders.')
check(math.isclose(delivered.gmv.sum(), s['gmv'], abs_tol=.001),
      'Delivered merchandise GMV reconciles to BRL 13,221,498.11.')
check(delivered.customer_unique_id.nunique() == s['customers'],
      'Unique delivered customers reconcile to 93,358.')
check(delivered.late.notna().sum() == s['delivery_eligible_n'] and
      delivered.late.sum() == s['late_orders'],
      'Lateness reconciles to 7,826 / 96,470 eligible delivered orders.')

nb = nbformat.read(ROOT / 'notebooks/olist_analysis.ipynb', as_version=4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == 'code']
check(len(code) == 12 and all(c.execution_count is not None for c in code) and
      not any(o.output_type == 'error' for c in code for o in c.outputs),
      'Notebook validates: all 12 code cells executed, zero error outputs.')
check(len(PdfReader(ROOT / 'reports/executive_summary.pdf').pages) == 1,
      'Executive summary contains exactly one PDF page.')
ppt = ROOT / 'reports/presentation.pptx'
with zipfile.ZipFile(ppt) as z:
    check(z.testzip() is None, 'PowerPoint archive passes CRC checks.')
    check(len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]) == 10,
          'Leadership PowerPoint contains exactly 10 slides.')
    check(len([n for n in z.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml', n)]) == 4,
          'PowerPoint contains four native chart parts.')
    check(len([n for n in z.namelist() if n.startswith('ppt/embeddings/') and n.endswith('.xlsx')]) == 4,
          'All four native charts include editable embedded workbooks.')

metrics = pd.read_csv(OUT / 'tables/model_test.csv')
b = r['selected_model']; selected = metrics.loc[metrics.selected.eq(True)].iloc[0]
check(selected.model == b['model'] and math.isclose(selected.average_precision, b['average_precision']),
      'Selected model and final test metrics reconcile across CSV and JSON.')
check(sum(b[k] for k in ('tn','fp','fn','tp')) == 6156,
      'Selected-model confusion matrix totals 6,156 July test orders.')
html = (ROOT / 'dashboard/index.html').read_text(encoding='utf-8')
check('__DATA__' not in html and '__PLOTLY__' not in html and 'plotly.js' in html.lower(),
      'Dashboard has embedded data and chart library; no unfilled placeholders.')

for description in checks:
    print('PASS:', description)
print(f'{len(checks)} submission checks passed.')
