from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Pandas is required
try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required (pip install pandas).", file=sys.stderr)
    raise

# Try to import slack_sdk; if missing, we'll transparently fallback to console print
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except Exception:
    SLACK_AVAILABLE = False


def load_scored_leads(csv_path: str) -> pd.DataFrame:
    """Load the scored leads CSV with expected columns: Lead_ID, Score, Probability."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"Lead_ID", "Score", "Probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Coerce types defensively
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    # Drop rows with invalid Score
    df = df.dropna(subset=["Score"])
    return df


def filter_hot_leads(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return only leads with Score > threshold, sorted descending by Score."""
    hot = df[df["Score"] > threshold].copy()
    hot = hot.sort_values(by=["Score", "Probability"], ascending=[False, False])
    return hot


def format_slack_message(lead_id, score, probability) -> str:
    """Format a clear message for sales reps."""
    # Probability shown with two decimals
    prob_str = f"{float(probability):.2f}" if pd.notna(probability) else "N/A"
    score_int = int(round(float(score)))
    return f"🚀 New Hot Lead: {lead_id}, Score {score_int} (p={prob_str}) → route to Sales immediately."


def send_slack_message(
    text: str,
    token: Optional[str],
    channel: Optional[str],
    dry_run_reason: Optional[str] = None,
) -> bool:
    """
    Send a Slack message if slack_sdk, token, and channel are available.
    Otherwise, print a safe console fallback. Returns True if 'sent' (or printed).
    """
    # If any required piece is missing, fallback to console
    if not SLACK_AVAILABLE:
        print(f"[FAKE SLACK] {text}  (reason: slack_sdk not installed)")
        return True

    if not token:
        print(f"[FAKE SLACK] {text}  (reason: SLACK_BOT_TOKEN not set)")
        return True

    if not channel:
        print(f"[FAKE SLACK] {text}  (reason: Slack channel not provided)")
        return True

    try:
        client = WebClient(token=token)
        client.chat_postMessage(channel=channel, text=text)
        print(f"[SLACK SENT] {text}")
        return True
    except SlackApiError as e:
        # On API failure, fallback to console so you still see the alert content
        print(f"[FAKE SLACK] {text}  (reason: SlackApiError {e.response.status_code}: {e.response.get('error')})")
        return True
    except Exception as e:
        print(f"[FAKE SLACK] {text}  (reason: {type(e).__name__}: {e})")
        return True


def notify_hot_leads(
    df: pd.DataFrame,
    threshold: float,
    token: Optional[str],
    channel: Optional[str],
) -> int:
    """Send or simulate Slack alerts for hot leads. Returns the number flagged."""
    hot = filter_hot_leads(df, threshold)
    for _, row in hot.iterrows():
        msg = format_slack_message(row["Lead_ID"], row["Score"], row.get("Probability"))
        send_slack_message(msg, token=token, channel=channel)

    # Summary stats
    total_flagged = len(hot)
    print("\n=== Summary ===")
    print(f"Hot threshold: > {threshold}")
    print(f"Total hot leads flagged: {total_flagged}")

    if total_flagged > 0:
        top5 = hot[["Lead_ID", "Score", "Probability"]].head(5)
        # Pretty print top 5
        print("\nTop 5 hot leads:")
        # Convert to a nice string table
        with pd.option_context("display.max_colwidth", 80):
            print(top5.to_string(index=False, formatters={
                "Score": lambda s: f"{int(round(s))}",
                "Probability": lambda p: f"{float(p):.2f}" if pd.notna(p) else "N/A",
            }))
    else:
        print("No leads exceeded the threshold.")

    return total_flagged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Slack alerts for hot leads based on scored_leads.csv"
    )
    parser.add_argument(
        "--csv",
        default="data/scored_leads.csv",
        help="Path to the scored leads CSV (default: data/scored_leads.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Score threshold for 'hot' leads (strictly greater than). Default: 80.0",
    )
    parser.add_argument(
        "--channel",
        default=os.environ.get("SLACK_CHANNEL"),
        help="Slack channel (e.g., '#lead-alerts'). Defaults to $SLACK_CHANNEL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Slack config (token via env var)
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_channel = args.channel  # may be None -> fallback printing

    try:
        df = load_scored_leads(args.csv)
    except Exception as e:
        print(f"ERROR loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    notify_hot_leads(df, threshold=args.threshold, token=slack_token, channel=slack_channel)


if __name__ == "__main__":
    main()