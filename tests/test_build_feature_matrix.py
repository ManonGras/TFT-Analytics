from features import build_feature_matrix as bfm


def test_extract_patch_finds_patch_in_older_set_version_string():
    version = "Linux Version 16.15.802.4387 (Aug 03 2026/19:42:24) [PUBLIC] <Releases/16.15>"
    assert bfm.extract_patch(version) == "16.15"


def test_extract_patch_falls_back_to_raw_string_when_no_digits():
    # Set 18's game_version is a Riot-side placeholder during the Unreal
    # engine transition: no digits to extract, so the whole string comes
    # back unchanged instead of crashing.
    version = "TFT Unreal Version ?.?.?.?"
    assert bfm.extract_patch(version) == version
