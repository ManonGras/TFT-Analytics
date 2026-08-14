import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURE_MATRIX_PATH = "data/processed/feature_matrix.csv"

# 16.15 and 16.16 are kept for testing only.
# This lets us see how well the model works on patches it has never seen.
TEST_PATCHES = {"16.15", "16.16"}

CATEGORICAL_COLS = ["seed_tier", "patch"]

# These features are removed because they are too closely related to the
# final placement. They describe what happened during the game, rather than
# the composition itself. Keeping them would make the model look better
# without really helping us understand which traits, items or champions
# are linked to placement.
#
# Their correlation with placement is quite high (0.51 to 0.89), so they
# would cause data leakage. gold_left is kept because its correlation is
# much lower (0.13) and it represents the player's economy.
DROP_COLS = [
    "participant_id", "match_id", "placement",
    "last_round", "time_eliminated", "players_eliminated",
    "total_damage_to_players", "level",
]

def patch_sort_key(patch):
    return tuple(int(p) for p in str(patch).split("."))


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
    df = pd.read_csv(FEATURE_MATRIX_PATH)

    is_test = df["patch"].astype(str).isin(TEST_PATCHES)
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
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows ({TEST_PATCHES})")
    patches_chrono = sorted(train_df["patch"].astype(str).unique(), key=patch_sort_key)
    print(f"  Train patches (chronological): {patches_chrono}")

    print("\nTraining HistGradientBoostingRegressor...")
    model = train(X_train, y_train)

    report(y_train, model.predict(X_train), "Train")
    pred_test = model.predict(X_test)
    report(y_test, pred_test, "Test (held-out future patches)")

    print("\n--- Test MAE by patch ---")
    test_df = test_df.copy()
    test_df["pred"] = pred_test
    for patch in sorted(test_df["patch"].astype(str).unique(), key=patch_sort_key):
        group = test_df[test_df["patch"].astype(str) == patch]
        print(f"  {patch}: MAE={mean_absolute_error(group['placement'], group['pred']):.3f} (n={len(group)})")

    print("\n--- Test MAE by tier ---")
    for tier, group in test_df.groupby("seed_tier"):
        print(f"  {tier}: MAE={mean_absolute_error(group['placement'], group['pred']):.3f} (n={len(group)})")


if __name__ == "__main__":
    main()
