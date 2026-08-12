import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm
from requests.exceptions import HTTPError

from riot_client import get_challenger_league, get_match_ids_by_puuid, get_match_by_id
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

def save_raw_match(match_id, match_json):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path= RAW_DIR / f"{match_id}.json"
    if not path.exists():
        path.write_text(json.dumps(match_json), encoding="utf-8")
    return path

def parse_match(match_json) -> Match:
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

def collect_seed_puuids(max_players):
    challenger = get_challenger_league()
    entries = challenger["entries"]
    puuids = [e["puuid"] for e in entries if "puuid" in e]
    return puuids[:max_players]

def collect_match_ids(puuids, matches_per_player):
    match_ids = set()
    for puuid in tqdm(puuids, desc="Retrieving matches list"):
        ids = get_match_ids_by_puuid(puuid, count=matches_per_player)
        match_ids.update(ids)
        time.sleep(REQUEST_DELAY_SECONDS)
    return match_ids

def ingest_matches(match_ids, session):
    already_id_db = {m.match_id for m in session.query(Match.match_id).all()}
    to_fetch = [m for m in match_ids if m not in already_id_db]

    print(f"Found {len(match_ids)} unique matches, {len(to_fetch)} to download "
         f"({len(match_ids) - len(to_fetch)} already in DB).")
    failures = []
    for match_id in tqdm(to_fetch, desc="Ingesting matches"):
        try:
            match_json = get_match_by_id(match_id)
            save_raw_match(match_id, match_json)
            match = parse_match(match_json)
            session.add(match)
            session.commit()
        except HTTPError as exc:
            session.rollback()
            failures.append((match_id, str(exc)))
        except Exception as exc:
            session.rollback()
            failures.append((match_id, f"parsing error: {exc}"))
        time.sleep(REQUEST_DELAY_SECONDS)

    if failures:
        print(f"\n{len(failures)}matches failed to ingest:")
        for match_id, reason in failures[:10]:
            print(f"  {match_id}: {reason}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more failures.")

def main():
    parser = argparse.ArgumentParser(description="Ingest TFT matches from the Riot API.")
    parser.add_argument("--players", type=int, default=10,
                         help="Number of Challenger players to use as seeds (default: 10).")
    parser.add_argument("--matches-per-player", type=int, default=20,
                         help="Number of recent matches per player (default: 20, API max: 20).")
    args = parser.parse_args()

    init_db()
    session = get_session()

    puuids = collect_seed_puuids(args.players)
    print(f"{len(puuids)} seed players selected.")

    match_ids = collect_match_ids(puuids, args.matches_per_player)
    ingest_matches(match_ids, session)

    session.close()
    print("Done.")


if __name__ == "__main__":
    main()