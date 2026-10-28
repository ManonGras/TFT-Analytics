import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import DATABASE_PATH
from db import get_engine, get_session, init_db
from ingest_matches import RAW_DIR, parse_match, skip_reason

# Rebuilds a database from the JSON files in data/raw, without any API call:
# for when the parsing (parse_match) or the schema changes. The raw files are
# the source of truth; the DB is only a parsed copy of them.
#
# seed_tier is not in the JSON (it records which ladder tier the match was
# found through), so it is copied from an existing database. Matches that
# database does not know get seed_tier = NULL.


def load_seed_tiers(db_path):
    """match_id -> seed_tier from an existing database ({} if it has none)."""
    if not Path(db_path).exists():
        return {}
    matches = pd.read_sql("SELECT match_id, seed_tier FROM matches", get_engine(db_path))
    return dict(zip(matches["match_id"], matches["seed_tier"]))


def reparse(raw_dir, session, seed_tiers):
    """Parse every raw file of the target set and queue into session. Returns
    counts of what was added and skipped."""
    counts = {"added": 0, "other_set": 0, "other_queue": 0, "no_seed_tier": 0, "unreadable": 0}
    for path in tqdm(sorted(Path(raw_dir).glob("*.json")), desc="Re-parsing raw matches"):
        try:
            match_json = json.loads(path.read_text(encoding="utf-8"))
            reason = skip_reason(match_json["info"])
        except (json.JSONDecodeError, KeyError):
            counts["unreadable"] += 1
            continue
        if reason is not None:
            counts[reason] += 1
            continue
        seed_tier = seed_tiers.get(match_json["metadata"]["match_id"])
        if seed_tier is None:
            counts["no_seed_tier"] += 1
        session.add(parse_match(match_json, seed_tier=seed_tier))
        counts["added"] += 1
        if counts["added"] % 1000 == 0:
            session.commit()
    session.commit()
    return counts


def main():
    parser = argparse.ArgumentParser(description="Rebuild a TFT database from data/raw.")
    parser.add_argument("--output", required=True,
                        help="Path of the new database (must not exist yet).")
    parser.add_argument("--seed-tiers-from", default=DATABASE_PATH,
                        help=f"Existing database to copy seed_tier from (default: {DATABASE_PATH}).")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    args = parser.parse_args()

    # Never write into an existing database: a re-parse is compared with the
    # current DB, it does not replace it.
    if Path(args.output).exists():
        parser.error(f"{args.output} already exists.")

    seed_tiers = load_seed_tiers(args.seed_tiers_from)
    print(f"seed_tier known for {len(seed_tiers)} matches (from {args.seed_tiers_from})")

    init_db(args.output)
    session = get_session(args.output)
    counts = reparse(args.raw_dir, session, seed_tiers)
    session.close()
    print(counts)


if __name__ == "__main__":
    main()
