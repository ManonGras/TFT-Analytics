import pandas as pd
import pytest

from modeling import train_model


def _fake_feature_matrix(n=100):
    # 8 consecutive rows per match, so all but the last match are full
    # 8-human lobbies (the last one has n % 8 humans).
    return pd.DataFrame({
        "participant_id": range(n),
        "match_id": [f"M{i // 8}" for i in range(n)],
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


def test_constant_categorical_column_is_not_declared_categorical(fake_matrix_path):
    # "patch" is constant in the fake matrix (as on Set 18); declaring it
    # categorical let it steal splits from seed_tier, so train() must skip it.
    X_train, y_train, _, _, _, _ = train_model.load_and_prepare()
    model = train_model.train(X_train, y_train)
    is_categorical = dict(zip(X_train.columns, model.is_categorical_))
    assert not is_categorical["patch"]
    assert is_categorical["seed_tier"]


def test_incomplete_lobbies_are_dropped_by_default(fake_matrix_path):
    _, _, _, _, train_df, test_df = train_model.load_and_prepare()
    kept = pd.concat([train_df, test_df])
    assert (kept.groupby("match_id").size() == train_model.FULL_LOBBY_SIZE).all()
    assert len(kept) == 96  # the 4-row match M12 is gone


def test_incomplete_lobbies_kept_on_request(fake_matrix_path):
    _, _, _, _, train_df, test_df = train_model.load_and_prepare(full_lobbies_only=False)
    assert len(train_df) + len(test_df) == 100


def test_top4_within_match_ranks_predictions_inside_each_lobby():
    # Predictions are all "bad" in absolute terms (>= 5), but their order
    # inside the lobby is right, so ranking gets every row right.
    y_true = [1, 2, 3, 4, 5, 6, 7, 8]
    y_pred = [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7]
    match_ids = ["A"] * 8
    assert train_model.top4_accuracy_within_match(y_true, y_pred, match_ids) == 1.0
    assert train_model.top4_accuracy_rounded(y_true, y_pred) == 0.5


def test_top4_within_match_does_not_mix_matches():
    y_true = [1, 8, 1, 8]
    y_pred = [1.0, 2.0, 7.0, 8.0]
    match_ids = ["A", "A", "B", "B"]
    # Per match, the better prediction is ranked first and is in the top 4;
    # with 2 players per match both are "top 4", so only placement 1s count.
    assert train_model.top4_accuracy_within_match(y_true, y_pred, match_ids) == 0.5


def test_ablation_table_has_one_row_per_model(fake_matrix_path):
    from modeling import baselines

    X_train, y_train, X_test, y_test, _, test_df = train_model.load_and_prepare()
    X_train["champ__A"], X_test["champ__A"] = 1, 2
    X_train["item__I"], X_test["item__I"] = 0, 1
    table = baselines.ablation_table(X_train, y_train, X_test, y_test, test_df["match_id"])
    assert len(table) == 6
    assert table["mae"].notna().all()


def test_top4_within_match_breaks_ties_at_random_not_by_row_order():
    # Rows are stored in placement order (as in the real DB). A constant
    # prediction must not inherit that order: expected accuracy is 50%.
    n_matches = 500
    y_true = list(range(1, 9)) * n_matches
    match_ids = [m for m in range(n_matches) for _ in range(8)]
    acc = train_model.top4_accuracy_within_match(y_true, [4.5] * len(y_true), match_ids)
    assert acc == pytest.approx(0.5, abs=0.03)


def test_train_works_without_a_categorical_column(fake_matrix_path):
    X_train, y_train, _, _, _, _ = train_model.load_and_prepare()
    model = train_model.train(X_train.drop(columns="seed_tier"), y_train)
    assert "seed_tier" not in model.feature_names_in_
