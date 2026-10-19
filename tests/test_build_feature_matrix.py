import pandas as pd

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


def _ms(timestamp):
    return int(pd.Timestamp(timestamp).value // 1_000_000)


CALENDAR = pd.DataFrame({
    "patch": ["18.1", "18.2"],
    "release_date": pd.to_datetime(["2026-08-26", "2026-09-10"]),
})
PLACEHOLDER = "TFT Unreal Version ?.?.?.?"


def test_patch_from_date_picks_the_patch_in_effect():
    dates = [_ms("2026-08-30 12:00"), _ms("2026-09-10 00:00"), _ms("2026-09-20")]
    assert bfm.patch_from_date(dates, CALENDAR).tolist() == ["18.1", "18.2", "18.2"]


def test_patch_from_date_is_nan_before_the_first_release():
    assert pd.isna(bfm.patch_from_date([_ms("2026-08-25 23:59")], CALENDAR).iloc[0])


def test_assign_patch_keeps_real_versions_and_infers_placeholders():
    versions = [
        "Linux Version 16.15.802.4387 (Aug 03 2026/19:42:24) [PUBLIC] <Releases/16.15>",
        PLACEHOLDER,
        PLACEHOLDER,
    ]
    dates = [_ms("2026-08-30"), _ms("2026-08-30"), _ms("2026-08-01")]
    # Real version wins over the date; a placeholder outside the calendar
    # stays as the raw string rather than being guessed.
    assert bfm.assign_patch(versions, dates, CALENDAR).tolist() == ["16.15", "18.1", PLACEHOLDER]


def test_patch_calendar_file_is_sorted_and_parsable():
    calendar = bfm.load_patch_calendar()
    assert calendar["release_date"].is_monotonic_increasing
    assert calendar["patch"].is_unique
    assert calendar["patch"].str.match(r"\d+\.\d+").all()
