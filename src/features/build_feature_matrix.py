import re
from pathlib import Path

import numpy as np
import pandas as pd

from db import get_engine
from set_paths import excluded_character_ids, feature_matrix_path, study_end_ms

# Riot fills empty slots in low-population lobbies with AI opponents, tagged
# with this literal puuid. They are not real players and must not appear in
# any composition/placement analysis.
BOT_PUUID = "BOT"

# Official patch release dates. Since Set 18 (Unreal engine), game_version is
# a placeholder ("TFT Unreal Version ?.?.?.?") with no patch number, so the
# patch is inferred from the match date instead. A patch starts at 00:00 UTC
# on its release date: matches played on a release day before the actual
# deployment are attributed to the new patch (Riot does not publish exact
# per-region deployment times).
PATCH_CALENDAR_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "tft_patch_calendar.csv"
PATCH_PATTERN = r"\d+\.\d+"


def extract_patch(game_version):
    match = re.search(r"(\d+\.\d+)\.", game_version)
    return match.group(1) if match else game_version


def load_patch_calendar(path=PATCH_CALENDAR_PATH):
    calendar = pd.read_csv(path, dtype={"patch": str}, parse_dates=["release_date"])
    return calendar.sort_values("release_date").reset_index(drop=True)


def patch_from_date(game_datetime_ms, calendar):
    """Patch in effect at each match date (game_datetime in ms, UTC). Dates
    before the first release in the calendar get NaN."""
    dates = pd.to_datetime(pd.Series(game_datetime_ms), unit="ms")
    idx = np.searchsorted(calendar["release_date"].to_numpy(), dates.to_numpy(), side="right") - 1
    patches = calendar["patch"].to_numpy()[np.clip(idx, 0, None)]
    return pd.Series(np.where(idx >= 0, patches, np.nan), index=dates.index, dtype=object)


def assign_patch(game_version, game_datetime_ms, calendar):
    """Patch from game_version when it has one, otherwise from the calendar.
    If neither works, the raw game_version string is kept so the problem
    stays visible instead of being silently filled."""
    patch = pd.Series(game_version).map(extract_patch)
    placeholder = ~patch.str.fullmatch(PATCH_PATTERN)
    inferred = patch_from_date(pd.Series(game_datetime_ms)[placeholder], calendar)
    patch[placeholder] = inferred.fillna(patch[placeholder])
    return patch


def board_counters(df):
    """Board-size counters derived from the composition columns. They carry no
    information about *which* units/traits/items are played, only how big the
    final board is, which mostly reflects how long the player survived. Used
    as a baseline: a composition model should beat them to claim it learned
    more than survival."""
    champ_cols = [c for c in df.columns if c.startswith("champ__")]
    trait_cols = [c for c in df.columns if c.startswith("trait__")]
    item_cols = [c for c in df.columns if c.startswith("item__")]
    return pd.DataFrame({
        "n_champ": (df[champ_cols] > 0).sum(axis=1),
        "sum_star": df[champ_cols].sum(axis=1),
        "n_items": df[item_cols].sum(axis=1),
        "n_traits": (df[trait_cols] > 0).sum(axis=1),
        "sum_trait_tier": df[trait_cols].sum(axis=1),
    }, index=df.index)


def load_base(engine, start_ms=None, end_ms=None):
    """Human participants, optionally restricted to start_ms <= game_datetime
    < end_ms (ms, UTC)."""
    date_filter = ""
    if start_ms is not None:
        date_filter += f" AND m.game_datetime >= {int(start_ms)}"
    if end_ms is not None:
        date_filter += f" AND m.game_datetime < {int(end_ms)}"
    df = pd.read_sql(
        f"""
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
            m.game_datetime,
            m.game_version
        FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        WHERE p.puuid != '{BOT_PUUID}'{date_filter}
        """,
        engine,
    )
    df["patch"] = assign_patch(df["game_version"], df["game_datetime"], load_patch_calendar())
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
    # A separate constant column to count, rather than reusing item_name as
    # both columns= and values=: pandas 2.2.3 (the version pinned for this
    # project) silently returns zero columns when the same column is used for
    # both, instead of raising an error.
    items["count"] = 1
    wide = items.pivot_table(
        index="participant_id", columns="item_name", values="count",
        aggfunc="sum", fill_value=0,
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
    units = units[~units["character_id"].isin(excluded_character_ids(engine))]
    wide = units.pivot_table(
        index="participant_id", columns="character_id", values="tier",
        aggfunc="max", fill_value=0,
    )
    wide.columns = [f"champ__{c}" for c in wide.columns]
    return wide


def build_matrix(engine, start_ms=None, end_ms=None, verbose=True):
    """One row per human participant in the date range, with the trait, item
    and champion columns. Feature columns that are all zero in the range (units
    or items that only exist in other patches) are dropped."""
    log = print if verbose else (lambda *args: None)
    log("Loading base participant data")
    base = load_base(engine, start_ms=start_ms, end_ms=end_ms)
    log(f"  {len(base)} participants")

    log("Loading trait features...")
    traits = load_trait_features(engine)
    log(f"  {traits.shape[1]} trait columns")

    log("Loading item features")
    items = load_item_features(engine)
    log(f"  {items.shape[1]} item columns")

    log("Loading champion features")
    champs = load_champion_features(engine)
    log(f"  {champs.shape[1]} champion columns")

    log("Merging")
    matrix = (
        base.set_index("participant_id")
        .join(traits, how="left")
        .join(items, how="left")
        .join(champs, how="left")
    )
    feature_cols = [c for c in matrix.columns if c.startswith(("trait__", "item__", "champ__"))]
    matrix[feature_cols] = matrix[feature_cols].fillna(0)
    empty_cols = [c for c in feature_cols if not matrix[c].any()]
    return matrix.drop(columns=empty_cols)


def main():
    engine = get_engine()
    matrix = build_matrix(engine, end_ms=study_end_ms())

    out_path = feature_matrix_path()
    matrix.to_csv(out_path)
    print(f"\nSaved {matrix.shape[0]} rows x {matrix.shape[1]} columns to {out_path}")


if __name__ == "__main__":
    main()
