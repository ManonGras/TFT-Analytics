import numpy as np
import pandas as pd

from db import get_engine
from features.build_feature_matrix import board_counters, build_matrix
from modeling.baselines import _fit_predict
from modeling.train_model import (
    FULL_LOBBY_SIZE, clip_placement, humans_per_match, load_and_prepare, prepare_categoricals, train,
)
from set_paths import set_suffixed_path, study_end_ms

N_BOOTSTRAP = 1000

# Same extra counters as the "full model + 3 counters" row of the ablation.
EXTRA_COUNTERS = ["n_champ", "sum_star", "n_items"]


def load_later_matches(engine, feature_cols):
    """Full lobbies played after the study window, with the training feature
    columns. Units/items the model never saw are dropped (it cannot use them)
    and returned so the loss is visible."""
    df = build_matrix(engine, start_ms=study_end_ms(), verbose=False).reset_index()
    df = df[humans_per_match(df) == FULL_LOBBY_SIZE].reset_index(drop=True)
    unseen = sorted(c for c in df.columns if c.startswith(("trait__", "item__", "champ__"))
                    and c not in feature_cols)
    return df, unseen


def bootstrap_mae(y_true, y_pred, match_ids, n_boot=N_BOOTSTRAP, random_state=0):
    """95% CI of the MAE, resampling whole matches (players of a match are
    not independent)."""
    errors = pd.DataFrame({"err": np.abs(np.asarray(y_true) - np.asarray(y_pred)),
                           "match": np.asarray(match_ids)})
    per_match = errors.groupby("match")["err"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(per_match), size=(n_boot, len(per_match)))
    maes = per_match[idx, 0].sum(axis=1) / per_match[idx, 1].sum(axis=1)
    return np.percentile(maes, [2.5, 97.5])


def later_patch_table(X_train, y_train, X_test, y_test, train_df, test_df, later_df,
                      random_state=42):
    """MAE of models trained on the study window, on its own test set and on
    each later patch. Compares the full model with the board-size counters."""
    X_later = later_df.reindex(columns=X_train.columns, fill_value=0)
    _, X_later = prepare_categoricals(train_df[X_train.columns], X_later)

    c_train, c_test, c_later = board_counters(X_train), board_counters(X_test), board_counters(X_later)
    models = {
        "full model": (train(X_train, y_train, random_state=random_state), X_test, X_later),
        "5 board-size counters": (None, c_test, c_later),
        "full model + 3 counters": (
            train(pd.concat([X_train, c_train[EXTRA_COUNTERS]], axis=1), y_train, random_state=random_state),
            pd.concat([X_test, c_test[EXTRA_COUNTERS]], axis=1),
            pd.concat([X_later, c_later[EXTRA_COUNTERS]], axis=1),
        ),
    }

    evaluation_sets = [("test (18.1, last days)", test_df, y_test, "test")]
    for patch, group in later_df.groupby("patch"):
        evaluation_sets.append((patch, group, group["placement"], "later"))

    rows = []
    for name, (model, X_eval_test, X_eval_later) in models.items():
        if model is None:
            pred_test = _fit_predict(c_train, y_train, c_test, random_state)
            pred_later = _fit_predict(c_train, y_train, c_later, random_state)
        else:
            pred_test, pred_later = model.predict(X_eval_test), model.predict(X_eval_later)
        pred_test, pred_later = clip_placement(pred_test), clip_placement(pred_later)
        pred_later = pd.Series(pred_later, index=later_df.index)
        for label, frame, y, which in evaluation_sets:
            pred = pred_test if which == "test" else pred_later.loc[frame.index].to_numpy()
            low, high = bootstrap_mae(y, pred, frame["match_id"])
            rows.append({
                "model": name, "evaluated_on": label, "n_matches": frame["match_id"].nunique(),
                "mae": float(np.mean(np.abs(np.asarray(y) - pred))), "ci_low": low, "ci_high": high,
            })
    return pd.DataFrame(rows)


def main():
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_prepare()
    later_df, unseen = load_later_matches(get_engine(), X_train.columns)
    print(f"Later full lobbies: {later_df['match_id'].nunique()} matches")
    print(f"Columns unknown to the model: {unseen}")
    table = later_patch_table(X_train, y_train, X_test, y_test, train_df, test_df, later_df)
    print(table.round(3).to_string(index=False))
    out_path = set_suffixed_path("later_patches", "csv")
    table.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
