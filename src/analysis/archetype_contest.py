import numpy as np
import pandas as pd

from features.build_feature_matrix import board_counters
from modeling.train_model import FULL_LOBBY_SIZE, humans_per_match
from set_paths import feature_matrix_path, set_suffixed_path

CLUSTERS_PATH = set_suffixed_path("archetype_clusters", "csv")

# Board-size controls: clusters are fitted on final boards, so they also carry
# how big the board got (i.e. how long the player survived). Without these
# controls, the contest coefficient mixes "being contested" with "having a
# small board".
BOARD_SIZE_CONTROLS = ["n_champ", "sum_star", "n_items"]


def load_data():
    return pd.read_csv(CLUSTERS_PATH)


def compute_contest_level(df):
    counts = (
        df.groupby(["match_id", "archetype_cluster"])
        .size()
        .rename("cluster_count_in_match")
        .reset_index()
    )
    df = df.merge(counts, on=["match_id", "archetype_cluster"], how="left")
    df["contest_level"] = df["cluster_count_in_match"] - 1  # exclude self
    return df


def add_board_counters(df):
    counters = board_counters(pd.read_csv(feature_matrix_path()).set_index("participant_id"))
    return df.merge(counters, left_on="participant_id", right_index=True, how="left")


def _ols(y, X, groups):
    """OLS coefficients with standard errors clustered by `groups` (players of
    the same match are not independent: their placements are a permutation of
    1-8)."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    scores = pd.DataFrame(X * (y - X @ beta)[:, None]).groupby(groups).sum().to_numpy()
    cov = xtx_inv @ (scores.T @ scores) @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def _demean_by(values, groups):
    return values - pd.DataFrame(values).groupby(groups).transform("mean").to_numpy()


def contest_regression(df, controls=(), match_fixed_effects=False, y=None):
    """Coefficient of `contest_level` on placement, with archetype fixed
    effects (each cluster's own baseline strength) and optional controls.
    With match fixed effects, only differences between players of the same
    lobby are used."""
    y = df["placement"].to_numpy(dtype=float) if y is None else np.asarray(y, dtype=float)
    dummies = pd.get_dummies(df["archetype_cluster"], prefix="cluster", dtype=float)
    if match_fixed_effects:
        # Demeaning within match absorbs the match effect; one cluster dummy
        # becomes redundant with the others and is dropped.
        dummies = dummies.iloc[:, 1:]
    X = np.column_stack([df[["contest_level", *controls]].to_numpy(dtype=float), dummies.to_numpy()])
    groups = df["match_id"].to_numpy()
    if match_fixed_effects:
        X, y = _demean_by(X, groups), _demean_by(y[:, None], groups)[:, 0]
    beta, se = _ols(y, X, groups)
    return {"coef": beta[0], "se": se[0], "n": len(y)}


def permute_within_match(placement, match_ids, rng):
    # Shuffles placements among players of the same match: keeps each
    # lobby's set of placements, breaks any link with contest.
    codes = pd.factorize(np.asarray(match_ids))[0]
    original_order = np.argsort(codes, kind="stable")
    shuffled_order = np.lexsort((rng.random(len(codes)), codes))
    permuted = np.empty(len(codes), dtype=float)
    permuted[original_order] = np.asarray(placement, dtype=float)[shuffled_order]
    return permuted


def permutation_null(df, n_permutations=100, random_state=42):
    """Contest coefficients obtained when placement is shuffled within each
    match. Shows what the coefficient looks like with no contest effect."""
    rng = np.random.default_rng(random_state)
    return np.array([
        contest_regression(df, y=permute_within_match(df["placement"], df["match_id"], rng))["coef"]
        for _ in range(n_permutations)
    ])


def eta_squared(values, groups):
    # Share of the variance of `values` explained by the group means.
    values = pd.Series(np.asarray(values, dtype=float))
    group_means = values.groupby(np.asarray(groups)).transform("mean")
    return float(((group_means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum())


def regression_table(df):
    specs = [
        ("cluster FE", (), False),
        ("cluster FE + board size", BOARD_SIZE_CONTROLS, False),
        ("cluster FE + match FE", (), True),
        ("cluster FE + match FE + board size", BOARD_SIZE_CONTROLS, True),
    ]
    rows = []
    for name, controls, match_fe in specs:
        rows.append({"specification": name, **contest_regression(df, controls, match_fe)})
    return pd.DataFrame(rows)


def bucket_contest(n):
    if n == 0:
        return "0 (uncontested)"
    if n == 1:
        return "1 other"
    return "2+ others"


def main():
    print("Loading archetype assignments")
    df = load_data()
    df = compute_contest_level(df)
    df["contest_bucket"] = df["contest_level"].apply(bucket_contest)

    print("\nContest level distribution:")
    print(df["contest_level"].value_counts().sort_index().to_string())

    print("\nOverall avg placement by contest level:")
    overall = df.groupby("contest_bucket")["placement"].agg(["mean", "count"])
    print(overall.round(3).to_string())

    print("\nPer-archetype avg placement by contest level:")
    summary = df.groupby(["archetype_cluster", "contest_bucket"]).agg(
        n=("participant_id", "count"),
        avg_placement=("placement", "mean"),
    ).reset_index()

    pivot_placement = summary.pivot(index="archetype_cluster", columns="contest_bucket", values="avg_placement")
    pivot_n = summary.pivot(index="archetype_cluster", columns="contest_bucket", values="n")
    cols = ["0 (uncontested)", "1 other", "2+ others"]
    pivot_placement = pivot_placement[[c for c in cols if c in pivot_placement.columns]]
    pivot_n = pivot_n[[c for c in cols if c in pivot_n.columns]]

    print("\nAvg placement:")
    print(pivot_placement.round(3).to_string())
    print("\nSample sizes:")
    print(pivot_n.to_string())

    pivot_placement["delta_uncontested_to_2plus"] = (
        pivot_placement.get("2+ others") - pivot_placement.get("0 (uncontested)")
    )
    print("\nDelta, uncontested -> 2+ contesting (positive = worse when contested):")
    print(pivot_placement["delta_uncontested_to_2plus"].sort_values(ascending=False).round(3).to_string())

    out_path = set_suffixed_path("archetype_contest", "csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    print(f"\n--- Is the contest effect more than board size? (full {FULL_LOBBY_SIZE}-human lobbies) ---")
    full = add_board_counters(df[humans_per_match(df) == FULL_LOBBY_SIZE].reset_index(drop=True))
    for col in BOARD_SIZE_CONTROLS:
        print(f"  eta² of {col} explained by archetype cluster: "
              f"{eta_squared(full[col], full['archetype_cluster']):.3f}")
    table = regression_table(full)
    print("\nContest coefficient (placement points per extra contesting player, SE clustered by match):")
    print(table.round(4).to_string(index=False))
    null = permutation_null(full)
    print(f"\nPlacements shuffled within match ({len(null)} permutations), cluster FE only: "
          f"coef mean={null.mean():.4f}, "
          f"95% range=[{np.percentile(null, 2.5):.4f}, {np.percentile(null, 97.5):.4f}]")
    out_path = set_suffixed_path("archetype_contest_regression", "csv")
    table.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()