from pathlib import Path

import pandas as pd

from config import DATABASE_PATH

# Some units are summons, PvE ennemies, etc and arent related to players strategic decisions
#  they shoudnt be treated as draftable champions.
EXCLUDED_CHARACTER_IDS_BY_SET = {
    17: (
        "TFT17_Summon",
        "TFT17_IvernMinion",
        "tft17_bardfollower",
        "TFT17_PVE_ElderDragon",
        "TFT17_Enemy_Aatrox",
    ),
    # Found once the dataset grew past ~135k participants: TFT18_SprykinSummonMelee is the
    # Sprykin trait's summoned companion (a real unit fielded on the board, not a player pick),
    # confirmed at 1 occurrence out of 314,774 participants. Revisit again if more turn up.
    18: (
        "TFT18_SprykinSummonMelee",
    ),
}


def excluded_character_ids(engine):
    set_numbers = pd.read_sql(
        "SELECT DISTINCT tft_set_number FROM matches", engine
    )["tft_set_number"].tolist()

    if len(set_numbers) != 1:
        print(f"  warning: expected exactly one tft_set_number in the DB, found {set_numbers}. "
              f"Not excluding any character_id.")
        return ()

    set_number = set_numbers[0]
    if set_number not in EXCLUDED_CHARACTER_IDS_BY_SET:
        print(f"  warning: no known EXCLUDED_CHARACTER_IDS for set {set_number}. "
              f"Not excluding any character_id.")
        return ()

    return EXCLUDED_CHARACTER_IDS_BY_SET[set_number]


def set_suffixed_path(base_name, ext):

    db_path = Path(DATABASE_PATH)
    if db_path.stem == "tft":
        return db_path.parent / f"{base_name}.{ext}"
    return db_path.parent / f"{base_name}_{db_path.stem}.{ext}"


def feature_matrix_path():
    return set_suffixed_path("feature_matrix", "csv")
