from db import get_session, Match, Participant
from ingest_matches import TARGET_SET_NUMBER


def main():
    session = get_session()

    stale_matches = (
        session.query(Match)
        .filter(Match.tft_set_number != TARGET_SET_NUMBER)
        .all()
    )
    if not stale_matches:
        print(f"No matches outside set {TARGET_SET_NUMBER}. Nothing to do.")
        return

    stale_ids = [m.match_id for m in stale_matches]
    participant_count = (
        session.query(Participant)
        .filter(Participant.match_id.in_(stale_ids))
        .count()
    )

    print(f"Found {len(stale_matches)} matches outside set {TARGET_SET_NUMBER} "
          f"({participant_count} participants, cascading to their units/traits/items/augments).")

    for match in stale_matches:
        session.delete(match)
    session.commit()

    print("Deleted.")


if __name__ == "__main__":
    main()
