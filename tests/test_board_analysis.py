import numpy as np
import pandas as pd
import pytest

from analysis import archetype_contest as contest
from analysis import archetypes
from features.build_feature_matrix import board_counters


def test_board_counters_counts_units_stars_items_and_traits():
    df = pd.DataFrame({
        "champ__A": [2, 0],
        "champ__B": [1, 3],
        "trait__X": [1, 0],
        "trait__Y": [2, 0],
        "item__I": [3, 1],
        "placement": [1, 8],  # ignored: not a composition column
    })
    counters = board_counters(df)
    assert counters["n_champ"].tolist() == [2, 1]
    assert counters["sum_star"].tolist() == [3, 3]
    assert counters["n_items"].tolist() == [3, 1]
    assert counters["n_traits"].tolist() == [2, 0]
    assert counters["sum_trait_tier"].tolist() == [3, 0]


def _synthetic_lobbies(contest_effect, n_matches=400, seed=0):
    # 8 players per match, 4 archetypes. Placement is driven by board size
    # plus `contest_effect` per contesting player, then turned into ranks.
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        clusters = rng.integers(0, 4, size=8)
        n_champ = rng.integers(3, 10, size=8)
        contest_level = np.array([(clusters == c).sum() - 1 for c in clusters])
        score = -n_champ + contest_effect * contest_level + rng.normal(0, 1, size=8)
        placement = score.argsort().argsort() + 1
        for i in range(8):
            rows.append({
                "match_id": f"M{m}", "archetype_cluster": clusters[i],
                "contest_level": contest_level[i], "n_champ": n_champ[i],
                "placement": placement[i],
            })
    return pd.DataFrame(rows)


def test_contest_regression_finds_a_real_effect():
    df = _synthetic_lobbies(contest_effect=2.0)
    result = contest.contest_regression(df, controls=["n_champ"])
    assert result["coef"] > 0
    assert result["coef"] / result["se"] > 5


def test_contest_regression_finds_no_effect_when_there_is_none():
    df = _synthetic_lobbies(contest_effect=0.0)
    for match_fe in (False, True):
        result = contest.contest_regression(df, controls=["n_champ"], match_fixed_effects=match_fe)
        assert abs(result["coef"]) < 3 * result["se"]


def test_permute_within_match_keeps_each_lobby_placements():
    df = _synthetic_lobbies(contest_effect=0.0, n_matches=20)
    permuted = contest.permute_within_match(df["placement"], df["match_id"], np.random.default_rng(1))
    original = df.groupby("match_id")["placement"].apply(sorted)
    shuffled = pd.Series(permuted, index=df.index).groupby(df["match_id"]).apply(sorted)
    assert (original == shuffled).all()
    assert not np.array_equal(permuted, df["placement"].to_numpy())


def test_eta_squared_bounds():
    assert contest.eta_squared([1, 1, 5, 5], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    assert contest.eta_squared([1, 5, 1, 5], ["a", "a", "b", "b"]) == pytest.approx(0.0)


def test_cluster_stability_is_high_for_well_separated_clusters():
    rng = np.random.default_rng(0)
    centers = np.eye(3) * 10
    X = pd.DataFrame(np.vstack([c + rng.normal(0, 0.1, size=(100, 3)) for c in centers]))
    result = archetypes.cluster_stability(X, k=3, n_runs=3, sample_size=200)
    assert result["ari_min"] > 0.99


def test_k20_is_in_the_scanned_candidates():
    assert 20 in archetypes.K_CANDIDATES
