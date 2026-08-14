import pandas as pd

from db import get_engine
from analysis.tier_comparison import load_participants, load_active_traits, load_items, summarize

LOW_TIERS = {"IRON", "BRONZE", "SILVER"}
HIGH_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER", "MASTER_PLUS"}

MIN_PICKS_PER_BUCKET = 30
TOP_N = 10


def bucket_of(tier):
    if tier in LOW_TIERS:
        return "LOW"
    if tier in HIGH_TIERS:
        return "HIGH"
    return None


def add_bucket(df):
    df = df.copy()
    df["seed_tier"] = df["seed_tier"].map(bucket_of)
    return df.dropna(subset=["seed_tier"])


def build_gap_table(summary_df):
    pivot = summary_df.pivot_table(
        index=summary_df.columns[1],  # trait_name or item_name
        columns="seed_tier",
        values=["pick_rate", "top4_rate", "picks"],
    )
    pivot.columns = [f"{value}_{bucket}" for value, bucket in pivot.columns]
    pivot = pivot.dropna(subset=["pick_rate_LOW", "pick_rate_HIGH"])
    pivot = pivot[
        (pivot["picks_LOW"] >= MIN_PICKS_PER_BUCKET)
        & (pivot["picks_HIGH"] >= MIN_PICKS_PER_BUCKET)
    ]
    pivot["pick_rate_gap"] = pivot["pick_rate_HIGH"] - pivot["pick_rate_LOW"]
    pivot["top4_rate_gap"] = pivot["top4_rate_HIGH"] - pivot["top4_rate_LOW"]
    return pivot


def print_gap_report(pivot, label):
    cols = ["pick_rate_LOW", "pick_rate_HIGH", "pick_rate_gap", "top4_rate_LOW", "top4_rate_HIGH"]

    print(f"\n=== {label}: picked more in HIGH tiers (Master+) than LOW tiers (Iron/Bronze/Silver) ===")
    print(pivot.sort_values("pick_rate_gap", ascending=False).head(TOP_N)[cols].to_string())

    print(f"\n=== {label}: picked more in LOW tiers than HIGH tiers ===")
    print(pivot.sort_values("pick_rate_gap").head(TOP_N)[cols].to_string())

    print(f"\n=== {label}: biggest top4-rate gap between skill levels (performs very differently by rank) ===")
    perf_cols = ["pick_rate_LOW", "pick_rate_HIGH", "top4_rate_LOW", "top4_rate_HIGH", "top4_rate_gap"]
    by_abs_gap = pivot.reindex(pivot["top4_rate_gap"].abs().sort_values(ascending=False).index)
    print(by_abs_gap.head(TOP_N)[perf_cols].to_string())


def main():
    engine = get_engine()

    participants = add_bucket(load_participants(engine))
    participants_per_bucket = participants.groupby("seed_tier")["participant_id"].count()
    print("Participants per bucket:")
    print(participants_per_bucket.to_string())

    traits = add_bucket(load_active_traits(engine))
    trait_summary = summarize(traits, participants_per_bucket, "trait_name")
    trait_gap = build_gap_table(trait_summary)
    print_gap_report(trait_gap, "Traits")
    trait_gap.to_csv("data/processed/trait_skill_gap.csv")

    items = add_bucket(load_items(engine))
    item_summary = summarize(items, participants_per_bucket, "item_name")
    item_gap = build_gap_table(item_summary)
    print_gap_report(item_gap, "Items")
    item_gap.to_csv("data/processed/item_skill_gap.csv")

    print("\nFull gap tables saved to data/processed/trait_skill_gap.csv and item_skill_gap.csv")


if __name__ == "__main__":
    main()
