import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from set_paths import feature_matrix_path

# The most recent TEST_FRACTION of matches (by game_datetime) are held out
# for testing. This lets us see how well the model works on games it has
# never seen, without relying on a patch identifier
TEST_FRACTION = 0.15

# Riot fills low-population lobbies with AI bots, which are dropped from the
# feature matrix. In those matches the humans' placements are not comparable
# to a full lobby (e.g. 4 humans + 4 bots: mean human placement about 2.56 instead
# of 4.5), so by default only matches with all 8 players human are kept.
FULL_LOBBY_SIZE = 8

CATEGORICAL_COLS = ["seed_tier", "patch"]

# These features are removed because they are too closely related to the
# final placement. They describe what happened during the game, rather than
# the composition itself. Keeping them would make the model look better
# without really helping us understand which traits, items or champions
# are linked to placement.
#
# Their correlation with placement is quite high (0.51 to 0.89), so they
# would cause data leakage. gold_left is kept because its correlation is
# much lower (0.13) and it represents the player's economy. game_datetime is
# only used to build the train/test split, not as a model feature.
DROP_COLS = [
    "participant_id", "match_id", "placement", "game_datetime",
    "last_round", "time_eliminated", "players_eliminated",
    "total_damage_to_players", "level",
]


def prepare_categoricals(X_train, X_test):
    # Encode categorical columns manually instead of using pandas' category
    # type. This keeps everything numeric, which is also good for SHAP.
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in CATEGORICAL_COLS:
        categories = sorted(X_train[col].dropna().unique())
        code_map = {cat: float(i) for i, cat in enumerate(categories)}

        X_train[col] = X_train[col].map(code_map)

        # If a category only appears in the test set, it becomes NaN.
        # This can happen with new patches that were not in the training data.
        # HistGradientBoostingRegressor can handle these missing values.
        X_test[col] = X_test[col].map(code_map)

    return X_train, X_test

def humans_per_match(df):
    # The feature matrix only holds human participants, so the row count of a
    # match is its number of humans.
    return df.groupby("match_id")["match_id"].transform("size")


def load_and_prepare(full_lobbies_only=True):
    df = pd.read_csv(feature_matrix_path())
    if full_lobbies_only:
        df = df[humans_per_match(df) == FULL_LOBBY_SIZE]

    cutoff = df["game_datetime"].quantile(1 - TEST_FRACTION)
    is_test = df["game_datetime"] >= cutoff
    train_df, test_df = df[~is_test].copy(), df[is_test].copy()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X_train, y_train = train_df[feature_cols], train_df["placement"]
    X_test, y_test = test_df[feature_cols], test_df["placement"]
    X_train, X_test = prepare_categoricals(X_train, X_test)

    return X_train, y_train, X_test, y_test, train_df, test_df


def train(X_train, y_train, random_state=42):
    # A categorical column with a single category (e.g. "patch" on Set 18,
    # where game_version carries no real patch number - see set_paths.py)
    # carries no information, but declaring it categorical to
    # HistGradientBoostingRegressor causes it to consume split slots that
    # should go to genuinely informative columns: confirmed by inspecting
    # the fitted trees, where "patch" absorbed real splits and seed_tier's
    # SHAP attribution collapsed to exactly zero as a result. Only declare
    # columns with real variation as categorical; a constant column left as
    # a plain numeric feature is correctly never split on.
    effective_categorical_cols = [
        c for c in CATEGORICAL_COLS if c in X_train.columns and X_train[c].nunique() > 1
    ]
    model = HistGradientBoostingRegressor(
        categorical_features=effective_categorical_cols, random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def top4_accuracy_rounded(y_true, y_pred):
    # Each prediction judged on its own: rounded predicted placement <= 4.
    rounded = np.clip(np.round(y_pred), 1, 8)
    return float(((np.asarray(y_true) <= 4) == (rounded <= 4)).mean())


def top4_accuracy_within_match(y_true, y_pred, match_ids, random_state=0):
    # Predictions ranked against the other players of the same match: the 4
    # best predicted placements of each lobby are called top 4. This respects
    # the fact that exactly 4 players per full lobby finish top 4.
    #
    # Ties are broken at random, never by row order: participants are stored
    # in placement order within each match (participant_id follows
    # placement), so a row-order tie-break would leak the answer. A constant
    # prediction would otherwise score 100%.
    y_pred = np.asarray(y_pred, dtype=float)
    codes = pd.factorize(np.asarray(match_ids))[0]
    tie_break = np.random.default_rng(random_state).random(len(y_pred))
    order = np.lexsort((tie_break, y_pred, codes))
    ranks = np.empty(len(y_pred))
    ranks[order] = pd.Series(codes[order]).groupby(codes[order]).cumcount().to_numpy() + 1
    return float(((np.asarray(y_true) <= 4) == (ranks <= 4)).mean())


def report(y_true, y_pred, label, match_ids=None):
    mae = mean_absolute_error(y_true, y_pred)
    baseline_mae = mean_absolute_error(y_true, np.full_like(y_pred, y_true.mean(), dtype=float))
    print(f"\n--- {label} (n={len(y_true)}) ---")
    print(f"  MAE: {mae:.3f}  (baseline, always predict mean placement: {baseline_mae:.3f})")
    print(f"  Top4 accuracy (rounded prediction): {top4_accuracy_rounded(y_true, y_pred):.3f}")
    if match_ids is not None:
        print(f"  Top4 accuracy (ranked within match): "
              f"{top4_accuracy_within_match(y_true, y_pred, match_ids):.3f}")


def main():
    print("Loading feature matrix...")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_prepare()
    train_dates = pd.to_datetime(train_df["game_datetime"], unit="ms")
    test_dates = pd.to_datetime(test_df["game_datetime"], unit="ms")
    print(f"  Train: {len(X_train)} rows ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"  Test: {len(X_test)} rows ({test_dates.min().date()} to {test_dates.max().date()})")

    print("\nTraining HistGradientBoostingRegressor...")
    model = train(X_train, y_train)

    report(y_train, model.predict(X_train), "Train", train_df["match_id"])
    pred_test = model.predict(X_test)
    report(y_test, pred_test, "Test (held-out future matches)", test_df["match_id"])

    print("\n--- Test MAE by day ---")
    test_df = test_df.copy()
    test_df["pred"] = pred_test
    test_df["day"] = test_dates.dt.date
    for day, group in test_df.groupby("day"):
        print(f"  {day}: MAE={mean_absolute_error(group['placement'], group['pred']):.3f} (n={len(group)})")

    print("\n--- Test MAE by tier ---")
    for tier, group in test_df.groupby("seed_tier"):
        print(f"  {tier}: MAE={mean_absolute_error(group['placement'], group['pred']):.3f} (n={len(group)})")


if __name__ == "__main__":
    main()
