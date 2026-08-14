import pandas as pd

from db import get_engine

MIN_PARTICIPANTS_FOR_STATS = 200

def load_participants(engine):
    return pd.read_sql(
        """
        SELECT p.id AS participant_id, p.placement, m.seed_tier
        FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        """,
        engine,
    )


def load_active_traits(engine):
    return pd.read_sql(
        """
        SELECT p.id AS participant_id, p.placement, m.seed_tier, t.trait_name
        FROM participant_traits t
        JOIN participants p ON p.id = t.participant_id
        JOIN matches m ON m.match_id = p.match_id
        WHERE t.tier_current > 0
        """,
        engine,
    )


def load_items(engine):
    return pd.read_sql(
        """
        SELECT p.id AS participant_id, p.placement, m.seed_tier, i.item_name
        FROM participant_unit_items i
        JOIN participant_units u ON u.id = i.unit_id
        JOIN participants p ON p.id = u.participant_id
        JOIN matches m ON m.match_id = p.match_id
        """,
        engine,
    )


def summarize(entity_df, participants_per_tier, entity_col):
    """Pick rate, average placement and top-4 rate per (tier, entity), deduped per participant."""
    unique = entity_df.drop_duplicates(subset=["participant_id", entity_col])

    grouped = unique.groupby(["seed_tier", entity_col]).agg(
        picks=("participant_id", "count"),
        avg_placement=("placement", "mean"),
        top4_rate=("placement", lambda s: (s <= 4).mean()),
    ).reset_index()

    grouped["tier_participants"] = grouped["seed_tier"].map(participants_per_tier)
    grouped["pick_rate"] = grouped["picks"] / grouped["tier_participants"]
    return grouped.drop(columns="tier_participants")


def print_top_per_tier(summary_df, entity_col, label, top_n=8, min_picks=20):
    reliable_tiers = summary_df.groupby("seed_tier")["picks"].sum()
    for tier in reliable_tiers.index:
        tier_df = summary_df[
            (summary_df["seed_tier"] == tier) & (summary_df["picks"] >= min_picks)
        ]
        top = tier_df.sort_values("pick_rate", ascending=False).head(top_n)
        print(f"\n--- {label} most picked in {tier} ---")
        print(top[[entity_col, "pick_rate", "avg_placement", "top4_rate"]].to_string(index=False))


def main():
    engine = get_engine()

    participants = load_participants(engine)
    participants_per_tier = participants.groupby("seed_tier")["participant_id"].count()

    print("Participants per tier:")
    print(participants_per_tier.sort_values(ascending=False).to_string())
    small_tiers = participants_per_tier[participants_per_tier < MIN_PARTICIPANTS_FOR_STATS]
    if not small_tiers.empty:
        print(f"\nWarning: low sample size for {list(small_tiers.index)} "
              f"(< {MIN_PARTICIPANTS_FOR_STATS} participants) - treat their rates as indicative only.")

    traits = load_active_traits(engine)
    trait_summary = summarize(traits, participants_per_tier, "trait_name")
    trait_summary.to_csv("data/processed/trait_summary_by_tier.csv", index=False)
    print_top_per_tier(trait_summary, "trait_name", "Traits")

    items = load_items(engine)
    item_summary = summarize(items, participants_per_tier, "item_name")
    item_summary.to_csv("data/processed/item_summary_by_tier.csv", index=False)
    print_top_per_tier(item_summary, "item_name", "Items")

    print("\nFull summaries saved to data/processed/trait_summary_by_tier.csv and item_summary_by_tier.csv")


if __name__ == "__main__":
    main()
