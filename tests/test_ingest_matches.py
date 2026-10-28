import json

import pandas as pd
import pytest
from sqlalchemy import text

import ingest_matches
import reparse_raw
from db import get_engine, get_session, init_db


def _fake_match(match_id="EUW1_1", set_number=ingest_matches.TARGET_SET_NUMBER,
                queue_key="queue_id", queue=ingest_matches.RANKED_QUEUE_ID):
    # Same shape as a TFT-MATCH-V1 response, reduced to the fields
    # parse_match reads. Participants are listed out of placement order on
    # purpose.
    participants = []
    for placement in [3, 1, 8, 2, 7, 4, 6, 5]:
        participants.append({
            "puuid": f"P{placement}",
            "placement": placement,
            "level": 10 - placement // 2,
            "last_round": 40 - placement,
            "players_eliminated": 0,
            "time_eliminated": 2000.0 - placement,
            "total_damage_to_players": 10 * placement,
            "gold_left": placement,
            "win": placement <= 4,
            "units": [
                {"character_id": "TFT18_A", "tier": 2, "rarity": 1,
                 "itemNames": ["Item_X", "Item_Y"]},
                {"character_id": "TFT18_B", "tier": 1, "rarity": 4},  # no itemNames key
            ],
            "traits": [{"name": "Trait_T", "num_units": 2, "style": 1,
                        "tier_current": 1, "tier_total": 3}],
            "augments": ["Aug_1", "Aug_2", "Aug_3"],
        })
    return {
        "metadata": {"match_id": match_id, "data_version": "6"},
        "info": {"game_datetime": 1_757_000_000_000, "game_length": 2000.0,
                 "game_version": "TFT Unreal Version ?.?.?.?", "tft_set_number": set_number,
                 "tft_set_core_name": "TFTSet18", "tft_game_type": "standard",
                 queue_key: queue, "participants": participants},
    }


def test_parse_match_keeps_every_participant_with_placements_1_to_8():
    match = ingest_matches.parse_match(_fake_match(), seed_tier="GOLD")
    assert len(match.participants) == 8
    assert sorted(p.placement for p in match.participants) == list(range(1, 9))
    assert match.seed_tier == "GOLD"


def test_parse_match_copies_fields_units_items_traits_augments():
    match = ingest_matches.parse_match(_fake_match())
    first = match.participants[0]  # placement 3
    assert (first.puuid, first.level, first.last_round, first.gold_left) == ("P3", 9, 37, 3)
    assert [u.character_id for u in first.units] == ["TFT18_A", "TFT18_B"]
    assert [i.item_name for i in first.units[0].items] == ["Item_X", "Item_Y"]
    assert first.units[1].items == []
    assert first.traits[0].trait_name == "Trait_T" and first.traits[0].tier_current == 1
    assert len(first.augments) == 3


def test_parse_match_reads_old_queueId_key():
    match = ingest_matches.parse_match(_fake_match(queue_key="queueId"))
    assert match.queue_id == ingest_matches.RANKED_QUEUE_ID


def test_skip_reason():
    assert ingest_matches.skip_reason(_fake_match()["info"]) is None
    assert ingest_matches.skip_reason(_fake_match(set_number=17)["info"]) == "other_set"
    assert ingest_matches.skip_reason(_fake_match(queue=1090)["info"]) == "other_queue"


def test_load_raw_match_returns_none_when_never_downloaded(tmp_path):
    assert ingest_matches.load_raw_match("EUW1_404", raw_dir=tmp_path) is None
    (tmp_path / "EUW1_1.json").write_text(json.dumps(_fake_match()), encoding="utf-8")
    assert ingest_matches.load_raw_match("EUW1_1", raw_dir=tmp_path)["metadata"]["match_id"] == "EUW1_1"


def test_parsed_match_stores_8_rows_per_match(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    session = get_session(db_path)
    session.add(ingest_matches.parse_match(_fake_match("EUW1_1")))
    session.add(ingest_matches.parse_match(_fake_match("EUW1_2")))
    session.commit()
    session.close()
    rows = pd.read_sql("SELECT match_id, placement FROM participants", get_engine(db_path))
    assert (rows.groupby("match_id").size() == 8).all()
    for _, placements in rows.groupby("match_id")["placement"]:
        assert sorted(placements) == list(range(1, 9))


def test_reparse_rebuilds_target_matches_and_keeps_seed_tier(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for match_json in [_fake_match("EUW1_1"), _fake_match("EUW1_2"),
                       _fake_match("EUW1_3", set_number=17), _fake_match("EUW1_4", queue=1090)]:
        (raw_dir / f"{match_json['metadata']['match_id']}.json").write_text(
            json.dumps(match_json), encoding="utf-8")
    (raw_dir / "EUW1_5.json").write_text("{not json", encoding="utf-8")

    db_path = tmp_path / "rebuilt.db"
    init_db(db_path)
    session = get_session(db_path)
    counts = reparse_raw.reparse(raw_dir, session, seed_tiers={"EUW1_1": "DIAMOND"})
    session.close()

    assert counts == {"added": 2, "other_set": 1, "other_queue": 1, "no_seed_tier": 1, "unreadable": 1}
    with get_engine(db_path).connect() as conn:
        tiers = dict(conn.execute(text("SELECT match_id, seed_tier FROM matches")).fetchall())
    assert tiers == {"EUW1_1": "DIAMOND", "EUW1_2": None}


def test_reparse_seed_tiers_from_missing_db_is_empty(tmp_path):
    assert reparse_raw.load_seed_tiers(tmp_path / "missing.db") == {}
