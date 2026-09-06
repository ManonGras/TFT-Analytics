import pandas as pd
import pytest

from modeling import train_model


def _fake_feature_matrix(n=100):
    return pd.DataFrame({
        "participant_id": range(n),
        "match_id": [f"M{i}" for i in range(n)],
        "placement": [(i % 8) + 1 for i in range(n)],
        "level": [9] * n,
        "gold_left": [0] * n,
        "last_round": [20] * n,
        "players_eliminated": [0] * n,
        "time_eliminated": [1000.0] * n,
        "total_damage_to_players": [50] * n,
        "seed_tier": ["GOLD" if i % 2 == 0 else "DIAMOND" for i in range(n)],
        "patch": ["18.1"] * n,
        "game_datetime": list(range(n)),  # strictly increasing -> chronological order
        "trait__Vanguard": [i % 3 for i in range(n)],
    })


@pytest.fixture
def fake_matrix_path(tmp_path, monkeypatch):
    path = tmp_path / "feature_matrix.csv"
    _fake_feature_matrix().to_csv(path, index=False)
    monkeypatch.setattr(train_model, "feature_matrix_path", lambda: path)
    return path


def test_split_is_strictly_chronological(fake_matrix_path):
    X_train, y_train, X_test, y_test, train_df, test_df = train_model.load_and_prepare()
    assert train_df["game_datetime"].max() <= test_df["game_datetime"].min()


def test_test_set_is_roughly_test_fraction_of_data(fake_matrix_path):
    _, _, _, _, train_df, test_df = train_model.load_and_prepare()
    total = len(train_df) + len(test_df)
    actual_fraction = len(test_df) / total
    assert actual_fraction == pytest.approx(train_model.TEST_FRACTION, abs=0.03)


def test_game_datetime_is_dropped_from_model_features(fake_matrix_path):
    X_train, _, X_test, _, _, _ = train_model.load_and_prepare()
    assert "game_datetime" not in X_train.columns
    assert "game_datetime" not in X_test.columns


def test_categorical_columns_are_numeric_after_prepare(fake_matrix_path):
    X_train, _, X_test, _, _, _ = train_model.load_and_prepare()
    for col in train_model.CATEGORICAL_COLS:
        assert pd.api.types.is_float_dtype(X_train[col])
        assert pd.api.types.is_float_dtype(X_test[col])


def test_prepare_categoricals_maps_test_only_category_to_nan():
    X_train = pd.DataFrame({"seed_tier": ["GOLD", "GOLD", "DIAMOND"]})
    X_test = pd.DataFrame({"seed_tier": ["GOLD", "MASTER"]})  # MASTER unseen in train

    original_cols = train_model.CATEGORICAL_COLS
    try:
        train_model.CATEGORICAL_COLS = ["seed_tier"]
        encoded_train, encoded_test = train_model.prepare_categoricals(X_train, X_test)
    finally:
        train_model.CATEGORICAL_COLS = original_cols

    assert encoded_test["seed_tier"].isna().sum() == 1
    assert encoded_train["seed_tier"].isna().sum() == 0
