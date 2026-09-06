import pandas as pd

from db import get_engine
from set_paths import set_suffixed_path
from analysis.tier_comparison import load_participants, load_active_traits, load_items, summarize

LOW_TIERS = {"IRON", "BRONZE", "SILVER"}
HIGH_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER", "MASTER_PLUS"}

MIN_PICKS_PER_BUCKET = 30
TOP_N = 10

ARCHETYPE_CLUSTERS_PATH = set_suffixed_path("archetype_clusters", "csv")


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


def load_archetype_assignments():
    """Each participant belongs to exactly one archetype cluster, this is
    a pick rate over whole compositions, not individual items, champions..."""
    return pd.read_csv(ARCHETYPE_CLUSTERS_PATH)[
        ["participant_id", "seed_tier", "placement", "archetype_cluster"]
    ]


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

    trait_path = set_suffixed_path("trait_skill_gap", "csv")
    item_path = set_suffixed_path("item_skill_gap", "csv")
    archetype_path = set_suffixed_path("archetype_skill_gap", "csv")

    traits = add_bucket(load_active_traits(engine))
    trait_summary = summarize(traits, participants_per_bucket, "trait_name")
    trait_gap = build_gap_table(trait_summary)
    print_gap_report(trait_gap, "Traits")
    trait_gap.to_csv(trait_path)

    items = add_bucket(load_items(engine))
    item_summary = summarize(items, participants_per_bucket, "item_name")
    item_gap = build_gap_table(item_summary)
    print_gap_report(item_gap, "Items")
    item_gap.to_csv(item_path)

    archetypes = add_bucket(load_archetype_assignments())
    archetype_summary = summarize(archetypes, participants_per_bucket, "archetype_cluster")
    archetype_gap = build_gap_table(archetype_summary)
    print_gap_report(archetype_gap, "Archetypes")
    archetype_gap.to_csv(archetype_path)

    print(f"\nFull gap tables saved to {trait_path}, {item_path}, and {archetype_path}")


if __name__ == "__main__":
    main()
