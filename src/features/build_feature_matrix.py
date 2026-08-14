import re

import pandas as pd

from db import get_engine


def extract_patch(game_version):
    match = re.search(r"(\d+\.\d+)\.", game_version)
    return match.group(1) if match else game_version


def load_base(engine):
    df = pd.read_sql(
        """
        SELECT
            p.id AS participant_id,
            p.match_id,
            p.placement,
            p.level,
            p.gold_left,
            p.last_round,
            p.players_eliminated,
            p.time_eliminated,
            p.total_damage_to_players,
            m.seed_tier,
            m.game_version
        FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        """,
        engine,
    )
    df["patch"] = df["game_version"].map(extract_patch)
    return df.drop(columns="game_version")


def load_trait_features(engine):
    traits = pd.read_sql(
        """
        SELECT participant_id, trait_name, tier_current
        FROM participant_traits
        WHERE tier_current > 0
        """,
        engine,
    )
    wide = traits.pivot_table(
        index="participant_id", columns="trait_name", values="tier_current",
        aggfunc="max", fill_value=0,
    )
    wide.columns = [f"trait__{c}" for c in wide.columns]
    return wide


def load_item_features(engine):
    items = pd.read_sql(
        """
        SELECT p.id AS participant_id, i.item_name
        FROM participant_unit_items i
        JOIN participant_units u ON u.id = i.unit_id
        JOIN participants p ON p.id = u.participant_id
        """,
        engine,
    )
    wide = items.pivot_table(
        index="participant_id", columns="item_name", values="item_name",
        aggfunc="count", fill_value=0,
    )
    wide.columns = [f"item__{c}" for c in wide.columns]
    return wide


def load_champion_features(engine):
    units = pd.read_sql(
        """
        SELECT participant_id, character_id, tier
        FROM participant_units
        """,
        engine,
    )
    wide = units.pivot_table(
        index="participant_id", columns="character_id", values="tier",
        aggfunc="max", fill_value=0,
    )
    wide.columns = [f"champ__{c}" for c in wide.columns]
    return wide


def main():
    engine = get_engine()

    print("Loading base participant data")
    base = load_base(engine)
    print(f"  {len(base)} participants")

    print("Loading trait features...")
    traits = load_trait_features(engine)
    print(f"  {traits.shape[1]} trait columns")

    print("Loading item features")
    items = load_item_features(engine)
    print(f"  {items.shape[1]} item columns")

    print("Loading champion features")
    champs = load_champion_features(engine)
    print(f"  {champs.shape[1]} champion columns")

    print("Merging")
    matrix = (
        base.set_index("participant_id")
        .join(traits, how="left")
        .join(items, how="left")
        .join(champs, how="left")
    )
    feature_cols = [c for c in matrix.columns if c.startswith(("trait__", "item__", "champ__"))]
    matrix[feature_cols] = matrix[feature_cols].fillna(0)

    out_path = "data/processed/feature_matrix.csv"
    matrix.to_csv(out_path)
    print(f"\nSaved {matrix.shape[0]} rows x {matrix.shape[1]} columns to {out_path}")


if __name__ == "__main__":
    main()
