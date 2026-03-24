import argparse
import numpy as np
import pandas as pd
from faker import Faker
import string

rng = np.random.default_rng(2025)
fake = Faker()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_sf_lead_id(n):
    # Salesforce Lead IDs typically start with '00Q' (object keyprefix) and are 18 chars when case-insensitive.
    # We'll generate 18-char IDs: '00Q' + 15 random [A-Z0-9].
    chars = np.array(list(string.ascii_uppercase + string.digits))
    ids = []
    for _ in range(n):
        tail = ''.join(rng.choice(chars, size=15))
        ids.append('00Q' + tail)
    return np.array(ids)

def main(n_rows=5000, out_path="data/leads.csv"):
    # Categorical spaces
    industries = np.array([
        "Software", "Healthcare", "Financial Services", "E-Commerce",
        "Manufacturing", "Education", "Nonprofit", "Logistics", "Energy"
    ])
    p_ind = np.array([0.22, 0.10, 0.13, 0.14, 0.12, 0.08, 0.07, 0.08, 0.06])

    lead_sources = np.array([
        "Web", "Phone Inquiry", "Partner Referral", "Purchased List",
        "Trade Show", "Web Download", "Email", "Other"
    ])
    p_src = np.array([0.28, 0.06, 0.12, 0.08, 0.08, 0.16, 0.17, 0.05])

    status_values = np.array([
        "Open - Not Contacted",
        "Working - Contacted",
        "Nurturing",
        "Qualified",
        "Unqualified",
        "Converted"
    ])
    # We'll assign initial non-converted statuses; converted set later.
    p_status_base = np.array([0.34, 0.28, 0.18, 0.10, 0.10, 0.00])

    # Draw categorical fields
    Industry = rng.choice(industries, size=n_rows, p=p_ind)
    Lead_Source = rng.choice(lead_sources, size=n_rows, p=p_src)
    base_Status = rng.choice(status_values, size=n_rows, p=p_status_base)

    # Generate names & companies
    Name = np.array([fake.name() for _ in range(n_rows)])
    Company = np.array([fake.company() for _ in range(n_rows)])

    # Annual Revenue by industry (lognormal parameters per industry to create realistic ranges)
    ind_mu_sigma = {
        "Software": (15.3, 1.0),         # exp(mu) ~ $4.5M median
        "Healthcare": (15.7, 1.1),
        "Financial Services": (16.0, 1.1),
        "E-Commerce": (15.4, 1.0),
        "Manufacturing": (16.1, 1.0),
        "Education": (14.6, 1.0),
        "Nonprofit": (14.2, 0.9),
        "Logistics": (15.8, 1.0),
        "Energy": (16.4, 1.1)
    }
    mu = np.array([ind_mu_sigma[i][0] for i in Industry])
    sigma = np.array([ind_mu_sigma[i][1] for i in Industry])
    Annual_Revenue = np.exp(rng.normal(mu, sigma))  # raw dollars
    # Clip to reasonable bounds (50k .. 5B)
    Annual_Revenue = np.clip(Annual_Revenue, 5e4, 5e9)

    # Engagement signals
    lam_pages = {
        "Web": 7.5, "Phone Inquiry": 3.0, "Partner Referral": 5.0, "Purchased List": 2.5,
        "Trade Show": 6.0, "Web Download": 9.0, "Email": 6.5, "Other": 3.5
    }
    Pages_Viewed = np.array([rng.poisson(lam_pages[s]) for s in Lead_Source])
    Pages_Viewed = np.clip(Pages_Viewed, 1, None)

    base_open = {
        "Web": 0.16, "Phone Inquiry": 0.10, "Partner Referral": 0.22, "Purchased List": 0.05,
        "Trade Show": 0.18, "Web Download": 0.26, "Email": 0.30, "Other": 0.12
    }
    Email_Open_Rate = np.array([base_open[s] for s in Lead_Source]) + rng.normal(0, 0.05, size=n_rows)
    Email_Open_Rate = np.clip(Email_Open_Rate, 0.01, 0.85)
    Email_Open_Rate = np.round(Email_Open_Rate, 3)

    cta_lambda = 0.4 * Pages_Viewed * Email_Open_Rate + rng.normal(0, 0.05, size=n_rows)
    cta_lambda = np.clip(cta_lambda, 0.01, None)
    CTA_Clicks = rng.poisson(cta_lambda)

    demo_bias = {
        "Web": 0.05, "Phone Inquiry": 0.08, "Partner Referral": 0.10, "Purchased List": 0.01,
        "Trade Show": 0.12, "Web Download": 0.20, "Email": 0.07, "Other": 0.03
    }
    demo_logit = (
        -2.0
        + np.array([demo_bias[s] for s in Lead_Source]) * 4.0
        + 0.15 * (Pages_Viewed - Pages_Viewed.mean()) / (Pages_Viewed.std() + 1e-6)
        + 0.25 * (CTA_Clicks - max(1, CTA_Clicks.mean())) / (CTA_Clicks.std() + 1e-6)
    )
    Demo_Requested = (rng.uniform(size=n_rows) < (1/(1+np.exp(-demo_logit)))).astype(int)

    # Propensity to convert to Opportunity
    pages_z = (Pages_Viewed - Pages_Viewed.mean()) / (Pages_Viewed.std() + 1e-6)
    clicks_z = (CTA_Clicks - CTA_Clicks.mean()) / (CTA_Clicks.std() + 1e-6)
    open_rate = Email_Open_Rate

    source_bias = {
        "Web": 0.10, "Phone Inquiry": 0.00, "Partner Referral": 0.25, "Purchased List": -0.25,
        "Trade Show": 0.18, "Web Download": 0.22, "Email": 0.12, "Other": -0.05
    }
    status_bias = {
        "Open - Not Contacted": -0.30,
        "Working - Contacted": -0.05,
        "Nurturing": -0.10,
        "Qualified": 0.25,
        "Unqualified": -0.60,
        "Converted": 0.60
    }
    industry_bias = {
        "Software": 0.18, "Healthcare": 0.10, "Financial Services": 0.15, "E-Commerce": 0.12,
        "Manufacturing": 0.05, "Education": 0.00, "Nonprofit": -0.10, "Logistics": 0.03, "Energy": 0.08
    }
    rev_bias = np.log10(np.clip(Annual_Revenue, 1e4, None)) - 5.0

    z = (
        -0.2
        + 0.55*pages_z
        + 0.75*open_rate
        + 0.65*clicks_z
        + 0.60*Demo_Requested
        + np.array([source_bias[s] for s in Lead_Source])
        + np.array([status_bias[s] for s in base_Status])
        + np.array([industry_bias[i] for i in Industry])
        + 0.10*rev_bias
        + rng.normal(0, 0.30, size=n_rows)
    )
    p_convert = sigmoid(z)

    # --- Calibrate overall conversion rate to a target ---
    target_rate = 0.20  # 20% feels realistic
    cutoff = np.quantile(p_convert, 1 - target_rate)
    Converted_to_Opportunity = (p_convert >= cutoff).astype(int)
    Status = base_Status.copy()
    Status[Converted_to_Opportunity == 1] = "Converted"
    promote_mask = (Converted_to_Opportunity == 0) & (Demo_Requested == 1) & (CTA_Clicks >= np.percentile(CTA_Clicks, 70))
    Status[promote_mask] = "Qualified"

    df = pd.DataFrame({
        "Lead_ID": make_sf_lead_id(n_rows),
        "Name": Name,
        "Company": Company,
        "Industry": Industry,
        "Annual_Revenue": np.round(Annual_Revenue, 2),
        "Lead_Source": Lead_Source,
        "Status": Status,
        "Pages_Viewed": Pages_Viewed.astype(int),
        "Email_Open_Rate": np.round(Email_Open_Rate, 3),
        "CTA_Clicks": CTA_Clicks.astype(int),
        "Demo_Requested": Demo_Requested.astype(int),
        "Converted_to_Opportunity": Converted_to_Opportunity.astype(int)
    })

    df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Salesforce-style Leads dataset.")
    parser.add_argument("--n", type=int, default=5000, help="Number of rows to generate (default: 5000)")
    parser.add_argument("--out", type=str, default="data/leads.csv", help="Output CSV path (default: data/leads.csv)")
    args = parser.parse_args()
    df = main(args.n, args.out)
    conv_rate = df["Converted_to_Opportunity"].mean()
    print(f"Wrote {len(df):,} rows to {args.out} | Conversion rate: {conv_rate:.2%}")
