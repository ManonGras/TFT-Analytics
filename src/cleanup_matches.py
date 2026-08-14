from sqlalchemy import text

from db import get_engine
from ingest_matches import TARGET_SET_NUMBER, RANKED_QUEUE_ID


STALE_MATCH_FILTER = "tft_set_number != :target_set OR queue_id != :ranked_queue"

CLEANUP_STATEMENTS = [
    f"""
    DELETE FROM participant_unit_items
    WHERE unit_id IN (
        SELECT pu.id FROM participant_units pu
        JOIN participants p ON p.id = pu.participant_id
        JOIN matches m ON m.match_id = p.match_id
        WHERE {STALE_MATCH_FILTER}
    )
    """,
    f"""
    DELETE FROM participant_units
    WHERE participant_id IN (
        SELECT p.id FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        WHERE {STALE_MATCH_FILTER}
    )
    """,
    f"""
    DELETE FROM participant_traits
    WHERE participant_id IN (
        SELECT p.id FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        WHERE {STALE_MATCH_FILTER}
    )
    """,
    f"""
    DELETE FROM participant_augments
    WHERE participant_id IN (
        SELECT p.id FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        WHERE {STALE_MATCH_FILTER}
    )
    """,
    f"""
    DELETE FROM participants
    WHERE match_id IN (
        SELECT match_id FROM matches WHERE {STALE_MATCH_FILTER}
    )
    """,
    f"DELETE FROM matches WHERE {STALE_MATCH_FILTER}",
]


def main():
    engine = get_engine()
    params = {"target_set": TARGET_SET_NUMBER, "ranked_queue": RANKED_QUEUE_ID}

    with engine.connect() as conn:
        stale_count = conn.execute(
            text(f"SELECT COUNT(*) FROM matches WHERE {STALE_MATCH_FILTER}"), params
        ).scalar()

    if not stale_count:
        print(f"No matches outside set {TARGET_SET_NUMBER} / queue {RANKED_QUEUE_ID}. Nothing to do.")
        return


    with engine.begin() as conn:
        for statement in CLEANUP_STATEMENTS:
            conn.execute(text(statement), params)

    print("Deleted.")


if __name__ == "__main__":
    main()
