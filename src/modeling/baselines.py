import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from features.build_feature_matrix import board_counters
from set_paths import set_suffixed_path
from modeling.train_model import (
    load_and_prepare, train, top4_accuracy_rounded, top4_accuracy_within_match,
)

COMPOSITION_PREFIXES = ("champ__", "trait__", "item__")

# Columns dropped from the model because they describe how long the player
# survived. If the composition columns can rebuild them, dropping them does
# not remove that information from the model.
SURVIVAL_TARGETS = ["level", "last_round", "time_eliminated"]


def _score(name, n_features, y_test, pred, match_ids):
    return {
        "model": name,
        "n_features": n_features,
        "mae": mean_absolute_error(y_test, pred),
        "top4_rounded": top4_accuracy_rounded(y_test, pred),
        "top4_within_match": top4_accuracy_within_match(y_test, pred, match_ids),
    }


def _fit_predict(X_train, y_train, X_test, random_state):
    model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def ablation_table(X_train, y_train, X_test, y_test, test_match_ids, random_state=42):
    """Compare the full model with baselines that only know the size of the
    board. If board-size counters alone match the full model, its accuracy
    cannot be credited to *which* champions/items/traits are played."""
    c_train, c_test = board_counters(X_train), board_counters(X_test)
    champ_cols = [c for c in X_train.columns if c.startswith("champ__")]
    rows = []

    rows.append(_score("always predict train mean", 0, y_test,
                       np.full(len(y_test), y_train.mean()), test_match_ids))

    rows.append(_score("n_items only", 1, y_test,
                       _fit_predict(c_train[["n_items"]], y_train, c_test[["n_items"]], random_state),
                       test_match_ids))

    rows.append(_score("5 board-size counters", c_train.shape[1], y_test,
                       _fit_predict(c_train, y_train, c_test, random_state), test_match_ids))

    rows.append(_score("champion presence only (binary)", len(champ_cols), y_test,
                       _fit_predict((X_train[champ_cols] > 0).astype(int), y_train,
                                    (X_test[champ_cols] > 0).astype(int), random_state),
                       test_match_ids))

    full = train(X_train, y_train, random_state=random_state)
    rows.append(_score("full model (project)", X_train.shape[1], y_test,
                       full.predict(X_test), test_match_ids))

    extra = ["n_champ", "sum_star", "n_items"]
    full_plus = train(pd.concat([X_train, c_train[extra]], axis=1), y_train, random_state=random_state)
    rows.append(_score("full model + 3 counters", X_train.shape[1] + len(extra), y_test,
                       full_plus.predict(pd.concat([X_test, c_test[extra]], axis=1)), test_match_ids))

    return pd.DataFrame(rows)


def survival_reconstruction(X_train, X_test, train_df, test_df, targets=SURVIVAL_TARGETS,
                            random_state=42):
    """Test-set R² of predicting each dropped survival column from the
    composition columns alone. A high R² means the leak those columns
    represent is still reachable through the composition features."""
    comp_cols = [c for c in X_train.columns if c.startswith(COMPOSITION_PREFIXES)]
    rows = []
    for target in targets:
        pred = _fit_predict(X_train[comp_cols], train_df[target], X_test[comp_cols], random_state)
        rows.append({"target": target, "r2_from_composition": r2_score(test_df[target], pred)})
    return pd.DataFrame(rows)


def main():
    print("Loading feature matrix...")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_prepare()
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\nAblation: full model vs board-size baselines (test set)")
    table = ablation_table(X_train, y_train, X_test, y_test, test_df["match_id"])
    print(table.round(3).to_string(index=False))
    out_path = set_suffixed_path("baseline_ablation", "csv")
    table.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    print("\nCan the composition columns rebuild the dropped survival columns? (test R²)")
    recon = survival_reconstruction(X_train, X_test, train_df, test_df)
    print(recon.round(3).to_string(index=False))
    out_path = set_suffixed_path("survival_reconstruction", "csv")
    recon.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
