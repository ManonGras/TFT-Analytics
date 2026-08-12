import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm
from requests.exceptions import HTTPError

from riot_client import (
    get_challenger_league,
    get_grandmaster_league,
    get_master_league,
    get_league_entries,
    get_match_ids_by_puuid,
    get_match_by_id,
)
from db import (
    init_db,
    get_session,
    Match,
    Participant,
    ParticipantUnit,
    ParticipantUnitItem,
    ParticipantTrait,
    ParticipantAugment,
)

RAW_DIR = Path("data/raw")
REQUEST_DELAY_SECONDS = 1.2
TARGET_SET_NUMBER = 17

# Tiers with no divisions: seeded directly from their dedicated endpoint.
APEX_TIERS = ["CHALLENGER", "GRANDMASTER", "MASTER"]
# Tiers split into divisions I-IV: seeded via the paginated entries endpoint.
DIVISION_TIERS = ["DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]
DIVISIONS = ["I", "II", "III", "IV"]

def save_raw_match(match_id, match_json):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{match_id}.json"
    if not path.exists():
        path.write_text(json.dumps(match_json), encoding="utf-8")
    return path

def parse_match(match_json, seed_tier=None) -> Match:
    info = match_json["info"]
    metadata = match_json["metadata"]

    match = Match(
        match_id=metadata["match_id"],
        data_version=metadata.get("data_version"),
        game_datetime=info.get("game_datetime"),
        game_length=info.get("game_length"),
        game_version=info.get("game_version"),
        tft_set_number=info.get("tft_set_number"),
        tft_set_core_name=info.get("tft_set_core_name"),
        tft_game_type=info.get("tft_game_type"),
        queue_id=info.get("queue_id", info.get("queueId")),
        seed_tier=seed_tier,
    )

    for p in info["participants"]:
        participant = Participant(
            puuid=p["puuid"],
            placement=p["placement"],
            level=p.get("level"),
            last_round=p.get("last_round"),
            players_eliminated=p.get("players_eliminated"),
            time_eliminated=p.get("time_eliminated"),
            total_damage_to_players=p.get("total_damage_to_players"),
            gold_left=p.get("gold_left"),
            win=p.get("win"),
        )

        for u in p.get("units", []):
            unit = ParticipantUnit(
                character_id=u["character_id"],
                tier=u.get("tier"),
                rarity=u.get("rarity"),
            )
            for item_name in u.get("itemNames", []):
                unit.items.append(ParticipantUnitItem(item_name=item_name))
            participant.units.append(unit)

        for t in p.get("traits", []):
            participant.traits.append(
                ParticipantTrait(
                    trait_name=t["name"],
                    num_units=t.get("num_units"),
                    style=t.get("style"),
                    tier_current=t.get("tier_current"),
                    tier_total=t.get("tier_total"),
                )
            )

        for augment_name in p.get("augments", []):
            participant.augments.append(ParticipantAugment(augment_name=augment_name))

        match.participants.append(participant)

    return match

def _take_unique(entries, seen, limit):
    puuids = []
    for entry in entries:
        puuid = entry.get("puuid")
        if not puuid or puuid in seen:
            continue
        seen.add(puuid)
        puuids.append(puuid)
        if len(puuids) >= limit:
            break
    return puuids

def collect_seed_puuids(players_per_tier):
    seen = set()
    seed_by_tier = {}

    seed_by_tier["CHALLENGER"] = _take_unique(get_challenger_league()["entries"], seen, players_per_tier)
    seed_by_tier["GRANDMASTER"] = _take_unique(get_grandmaster_league()["entries"], seen, players_per_tier)
    seed_by_tier["MASTER"] = _take_unique(get_master_league()["entries"], seen, players_per_tier)

    for tier in DIVISION_TIERS:
        collected = []
        for division in DIVISIONS:
            if len(collected) >= players_per_tier:
                break
            page = 1
            while len(collected) < players_per_tier:
                entries = get_league_entries(tier, division, page)
                if not entries:
                    break
                collected.extend(_take_unique(entries, seen, players_per_tier - len(collected)))
                page += 1
        seed_by_tier[tier] = collected

    return seed_by_tier

def collect_match_ids(seed_by_tier, matches_per_player):
    match_tier = {}
    for tier, puuids in seed_by_tier.items():
        for puuid in tqdm(puuids, desc=f"Retrieving matches list ({tier})"):
            for match_id in get_match_ids_by_puuid(puuid, count=matches_per_player):
                match_tier.setdefault(match_id, tier)
            time.sleep(REQUEST_DELAY_SECONDS)
    return match_tier

def ingest_matches(match_tier, session):
    already_in_db = {m.match_id for m in session.query(Match.match_id).all()}
    to_fetch = {mid: tier for mid, tier in match_tier.items() if mid not in already_in_db}

    print(f"Found {len(match_tier)} unique matches, {len(to_fetch)} to download "
         f"({len(match_tier) - len(to_fetch)} already in DB).")
    skipped_other_set = 0
    failures = []
    for match_id, tier in tqdm(to_fetch.items(), desc="Ingesting matches"):
        try:
            match_json = get_match_by_id(match_id)
            save_raw_match(match_id, match_json)

            if match_json["info"].get("tft_set_number") != TARGET_SET_NUMBER:
                skipped_other_set += 1
                continue

            match = parse_match(match_json, seed_tier=tier)
            session.add(match)
            session.commit()
        except HTTPError as exc:
            session.rollback()
            failures.append((match_id, str(exc)))
        except Exception as exc:
            session.rollback()
            failures.append((match_id, f"parsing error: {exc}"))
        time.sleep(REQUEST_DELAY_SECONDS)

    if skipped_other_set:
        print(f"\nSkipped {skipped_other_set} matches from a different set "
              f"(target: set {TARGET_SET_NUMBER}).")

    if failures:
        print(f"\n{len(failures)}matches failed to ingest:")
        for match_id, reason in failures[:10]:
            print(f"  {match_id}: {reason}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more failures.")

def main():
    parser = argparse.ArgumentParser(description="Ingest TFT matches from the Riot API.")
    parser.add_argument("--players-per-tier", type=int, default=5,
                         help="Number of seed players per rank tier (default: 5).")
    parser.add_argument("--matches-per-player", type=int, default=20,
                         help="Number of recent matches per player (default: 20, API max: 20).")
    args = parser.parse_args()

    init_db()
    session = get_session()

    seed_by_tier = collect_seed_puuids(args.players_per_tier)
    for tier, puuids in seed_by_tier.items():
        print(f"  {tier}: {len(puuids)} seed players")

    match_tier = collect_match_ids(seed_by_tier, args.matches_per_player)
    ingest_matches(match_tier, session)

    session.close()
    print("Done.")


if __name__ == "__main__":
    main()