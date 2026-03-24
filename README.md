# Lead Scoring Automation System

A three-stage lead scoring pipeline that combines rule-based logic and logistic regression to prioritize B2B leads by conversion probability — modeled to project a **12% MQL→SQL conversion lift** — with automated Slack alerts for hot leads.

Built to replicate the kind of scoring infrastructure a RevOps or Marketing Ops team would run on top of a CRM like Salesforce.

---

## How it works

The system runs as three sequential scripts ("pods"):

```
Pod 1: generate_salesforce_leads.py   →   leads.csv (5,000 rows)
Pod 2: score_leads.py                 →   scored_leads.csv (score + probability per lead)
Pod 3: slack_alerts.py                →   Slack alerts for leads above threshold
```

### Pod 1 — Synthetic Data Generation

Generates 5,000 Salesforce-style leads with realistic behavioral and firmographic signals:

- **Behavioral**: Pages viewed, email open rate, CTA clicks, demo requested
- **Firmographic**: Industry (9 types), annual revenue (lognormal by industry), lead source (8 types)
- Engagement signals are source-conditional — "Web Download" leads have higher open rates and page views than "Purchased List" leads, matching real-world patterns
- Conversion probability uses a sigmoid function across 8+ feature interactions, calibrated to a ~20% overall conversion rate

### Pod 2 — Scoring Pipeline

Runs a **dual-model** approach on the lead dataset:

**Rule-based score (0–100)**
Assigns points for engagement signals and firmographic fit:
- Pages viewed, CTA clicks, email open rate, demo requested, annual revenue > $1M
- Source and industry bonuses/penalties (e.g., Purchased List: −15, Software industry: +10)
- Normalized to 0–100

**ML score (logistic regression)**
Trained on the same dataset with an 80/20 stratified split:
- Features: 5 numerical (revenue, pages, open rate, CTAs, demo) + 3 categorical (industry, source, status)
- Preprocessing: `StandardScaler` for numerical features, `OneHotEncoder` for categoricals
- Balanced class weights to handle the ~20% conversion rate imbalance
- Outputs probability of conversion (0–1)

Both scores are written to `data/scored_leads.csv` alongside the original lead data.

### Pod 3 — Slack Alerts

Loads scored leads and sends alerts to a Slack channel for any lead above a configurable threshold (default: 80). Gracefully falls back to console output if no token is configured.

---

## Setup

```bash
git clone https://github.com/romanlicursi/lead-scoring-automation
cd lead-scoring-automation
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

**Step 1 — Generate leads**
```bash
python scripts/generate_salesforce_leads.py --n 5000 --out data/leads.csv
# → Wrote 5,000 rows to data/leads.csv | Conversion rate: 20.00%
```

**Step 2 — Score leads**
```bash
python scripts/score_leads.py
# → Rows: 5,000 | Mean score | LogReg AUC | Top 5 leads
```

**Step 3 — Alert on hot leads**
```bash
# With Slack
export SLACK_BOT_TOKEN="xoxb-your-token"
export SLACK_CHANNEL="#lead-alerts"
python scripts/slack_alerts.py --threshold 80.0

# Console-only (no Slack token needed)
python scripts/slack_alerts.py --csv data/scored_leads.csv --threshold 80.0
```

---

## Results

| Metric | Value |
|---|---|
| Leads scored | 5,000 |
| Conversion rate (calibrated) | ~20% |
| Projected MQL→SQL lift | 12% |
| Rule-based score range | 0–100 |
| ML model | Logistic Regression |
| Default hot-lead threshold | Score > 80 |

---

## Project structure

```
lead-scoring-automation/
├── data/
│   ├── leads.csv              # Generated input (5,000 leads)
│   └── scored_leads.csv       # Output with rule score + ML probability
├── scripts/
│   ├── generate_salesforce_leads.py   # Pod 1: synthetic CRM data
│   ├── score_leads.py                 # Pod 2: dual-model scoring pipeline
│   └── slack_alerts.py                # Pod 3: threshold alerts
├── requirements.txt
└── .env                       # SLACK_BOT_TOKEN, SLACK_CHANNEL
```

---

## Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Slack SDK` · `Faker`
