import json
from pathlib import Path

from riot_client import (
    get_challenger_league,
    get_summoner_by_id,
    get_match_ids_by_puuid,
    get_match_by_id,
)


def show(label, response):
    print(f"\n--- {label} ---")
    print("status:", response.status_code)
    print("rate limit headers:", {
        k: v for k, v in response.headers.items() if "rate-limit" in k.lower()
    })
    if response.status_code != 200:
        print("body:", response.text[:500])
        return None
    return response.json()


def main():
    # 1. Challenger ladder (platform routing)
    challenger = show("challenger league", get_challenger_league())
    if challenger is None:
        return

    entries = challenger["entries"]
    print(f"number of challenger players: {len(entries)}")
    first_entry = entries[0]
    print("keys available on an entry:", list(first_entry.keys()))

    # 2. Resolve summonerId -> puuid (platform routing)
    # First check if puuid is already present directly in the ladder entry
    # (Riot has evolved some endpoints to include it natively).
    if "puuid" in first_entry:
        puuid = first_entry["puuid"]
        print("puuid found directly in the ladder entry, no resolution needed.")
    else:
        summoner = show("summoner by id", get_summoner_by_id(first_entry["summonerId"]))
        if summoner is None:
            return
        puuid = summoner["puuid"]

    print("puuid resolved:", puuid)

    # 3. Match ids (routage regional)
    match_ids = show("match ids by puuid", get_match_ids_by_puuid(puuid, count=5))
    if not match_ids:
        return
    print("match ids:", match_ids)

    # 4. Match detail (regional routing)
    match = show("match detail", get_match_by_id(match_ids[0]))
    if match is None:
        return

    print("top-level keys of the match:", list(match.keys()))
    print("keys of match['info']:", list(match["info"].keys()))
    print("keys of a participant:", list(match["info"]["participants"][0].keys()))

    # Save for detailed manual inspection
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{match_ids[0]}.json"
    out_path.write_text(json.dumps(match, indent=2), encoding="utf-8")
    print(f"\nFull match saved to {out_path} for manual inspection.")


if __name__ == "__main__":
    main()