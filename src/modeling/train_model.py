import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from set_paths import feature_matrix_path

# The most recent TEST_FRACTION of matches (by game_datetime) are held out
# for testing. This lets us see how well the model works on games it has
# never seen, without relying on a patch identifier
TEST_FRACTION = 0.15

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

def load_and_prepare():
    df = pd.read_csv(feature_matrix_path())

    cutoff = df["game_datetime"].quantile(1 - TEST_FRACTION)
    is_test = df["game_datetime"] >= cutoff
    train_df, test_df = df[~is_test].copy(), df[is_test].copy()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X_train, y_train = train_df[feature_cols], train_df["placement"]
    X_test, y_test = test_df[feature_cols], test_df["placement"]
    X_train, X_test = prepare_categoricals(X_train, X_test)

    return X_train, y_train, X_test, y_test, train_df, test_df


def train(X_train, y_train):
    model = HistGradientBoostingRegressor(categorical_features=CATEGORICAL_COLS, random_state=42)
    model.fit(X_train, y_train)
    return model


def report(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    baseline_mae = mean_absolute_error(y_true, np.full_like(y_pred, y_true.mean(), dtype=float))
    rounded = np.clip(np.round(y_pred), 1, 8)
    top4_acc = ((y_true <= 4) == (rounded <= 4)).mean()
    print(f"\n--- {label} (n={len(y_true)}) ---")
    print(f"  MAE: {mae:.3f}  (baseline, always predict mean placement: {baseline_mae:.3f})")
    print(f"  Top4 accuracy: {top4_acc:.3f}")


def main():
    print("Loading feature matrix...")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_prepare()
    train_dates = pd.to_datetime(train_df["game_datetime"], unit="ms")
    test_dates = pd.to_datetime(test_df["game_datetime"], unit="ms")
    print(f"  Train: {len(X_train)} rows ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"  Test: {len(X_test)} rows ({test_dates.min().date()} to {test_dates.max().date()})")

    print("\nTraining HistGradientBoostingRegressor...")
    model = train(X_train, y_train)

    report(y_train, model.predict(X_train), "Train")
    pred_test = model.predict(X_test)
    report(y_test, pred_test, "Test (held-out future matches)")

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
