from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

import set_paths


def test_set_suffixed_path_keeps_legacy_name_for_original_set17_db(monkeypatch):
    monkeypatch.setattr(set_paths, "DATABASE_PATH", "C:/data/processed/tft.db")
    result = set_paths.set_suffixed_path("feature_matrix", "csv")
    assert result == Path("C:/data/processed/feature_matrix.csv")


def test_set_suffixed_path_suffixes_other_dbs(monkeypatch):
    monkeypatch.setattr(set_paths, "DATABASE_PATH", "C:/data/processed/tft_set18.db")
    result = set_paths.set_suffixed_path("feature_matrix", "csv")
    assert result == Path("C:/data/processed/feature_matrix_tft_set18.csv")


def test_set_suffixed_path_never_collides_between_two_different_dbs(monkeypatch):
    monkeypatch.setattr(set_paths, "DATABASE_PATH", "C:/data/processed/tft.db")
    set17_path = set_paths.set_suffixed_path("archetype_clusters", "csv")

    monkeypatch.setattr(set_paths, "DATABASE_PATH", "C:/data/processed/tft_set18.db")
    set18_path = set_paths.set_suffixed_path("archetype_clusters", "csv")

    assert set17_path != set18_path


def _engine_with_matches(set_numbers):
    engine = create_engine("sqlite:///:memory:")
    pd.DataFrame({"tft_set_number": set_numbers}).to_sql("matches", engine, index=False)
    return engine


def test_excluded_character_ids_returns_set17_list_for_set17_db():
    engine = _engine_with_matches([17, 17, 17])
    assert set_paths.excluded_character_ids(engine) == set_paths.EXCLUDED_CHARACTER_IDS_BY_SET[17]


def test_excluded_character_ids_returns_empty_for_set18_db():
    engine = _engine_with_matches([18, 18])
    assert set_paths.excluded_character_ids(engine) == ()


def test_excluded_character_ids_warns_and_returns_empty_for_unknown_set(capsys):
    engine = _engine_with_matches([99])
    result = set_paths.excluded_character_ids(engine)
    assert result == ()
    assert "warning" in capsys.readouterr().out.lower()


def test_excluded_character_ids_warns_and_returns_empty_when_sets_are_mixed(capsys):
    # cleanup_matches.py is supposed to keep exactly one set per DB; if that
    # invariant is ever broken, fail safe instead of silently reusing another
    # set's exclusion list against unrelated character_ids.
    engine = _engine_with_matches([17, 18])
    result = set_paths.excluded_character_ids(engine)
    assert result == ()
    assert "warning" in capsys.readouterr().out.lower()
