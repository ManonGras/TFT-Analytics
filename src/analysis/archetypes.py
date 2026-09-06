import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from set_paths import feature_matrix_path, set_suffixed_path

FEATURE_MATRIX_PATH = feature_matrix_path()
K_CANDIDATES = [8, 10, 12, 15, 18, 22, 26, 30]
SCAN_SAMPLE_SIZE = 15000
RANDOM_STATE = 42


def load_composition_matrix():
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    comp_cols = [c for c in df.columns if c.startswith(("trait__", "champ__"))]
    X = (df[comp_cols] > 0).astype(int)
    return df, X, comp_cols


def scan_k(X, k_candidates=K_CANDIDATES, sample_size=SCAN_SAMPLE_SIZE, random_state=RANDOM_STATE):
    sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    results = []
    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(sample)
        sil = silhouette_score(sample, labels, sample_size=5000, random_state=random_state)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k:2d}  inertia={km.inertia_:10.1f}  silhouette={sil:.4f}")
    return pd.DataFrame(results)


def fit_final(X, k, random_state=RANDOM_STATE):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def cluster_signature(X, labels, cluster_id, top_n=8):
    """Traits/champions most over-represented in this cluster vs. the overall
    population (lift = cluster rate / overall rate). This is what lets us
    recognize and name a cluster as a real, known comp."""
    in_cluster = X[labels == cluster_id]
    overall_rate = X.mean()
    cluster_rate = in_cluster.mean()
    lift = (cluster_rate / overall_rate.replace(0, np.nan)).fillna(0)
    return lift.sort_values(ascending=False).head(top_n)


def summarize_clusters(df, X, comp_cols, labels, top_n=8):
    rows = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        sig = cluster_signature(X, labels, cluster_id, top_n=top_n)
        rows.append({
            "cluster": cluster_id,
            "n": mask.sum(),
            "avg_placement": df.loc[mask, "placement"].mean(),
            "signature": ", ".join(sig.index.str.replace("^(trait__|champ__)", "", regex=True)),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def describe_cluster(df, X, raw, labels, cluster_id, min_freq=0.20, top_n_traits=10, top_n_champs=12):
    """Full board description, not just what's distinctive. Champions are
    ranked by frequency within the cluster (how often they actually appear on
    these boards) rather than by lift ; lift alone only surfaces what
    differentiates this cluster from others, which earlier left out generic
    but very real core units (e.g a cluster's frontline tank shared with
    several other archetypes has low lift but is still on most of its
    boards). Lift is kept alongside each entry so a defining pick (high freq
    + high lift) can still be told apart from shared filler (high freq, lift
    near 1). Champion rows also show the average star level, to tell a carry
    (2-3 star) from a flex (1 star)."""
    mask = labels == cluster_id
    in_cluster = X[mask]
    overall_rate = X.mean()
    cluster_rate = in_cluster.mean()
    lift = (cluster_rate / overall_rate.replace(0, np.nan)).fillna(0)

    trait_cols = [c for c in cluster_rate.index if c.startswith("trait__")]
    champ_cols = [c for c in cluster_rate.index if c.startswith("champ__")]

    traits = []
    for col in cluster_rate[trait_cols].sort_values(ascending=False).index[:top_n_traits]:
        if cluster_rate[col] < min_freq:
            break
        traits.append({"name": col.replace("trait__", ""), "freq": cluster_rate[col], "lift": lift[col]})

    champs = []
    for col in cluster_rate[champ_cols].sort_values(ascending=False).index[:top_n_champs]:
        if cluster_rate[col] < min_freq:
            break
        levels = raw.loc[mask, col]
        champs.append({
            "name": col.replace("champ__", ""),
            "freq": cluster_rate[col],
            "lift": lift[col],
            "avg_star": levels[levels > 0].mean(),
        })
    return traits, champs


def print_cluster_reports(df, X, raw, labels):
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        traits, champs = describe_cluster(df, X, raw, labels, cluster_id)
        print(f"\n=== Cluster {cluster_id} - n={mask.sum()}, avg_placement={df.loc[mask, 'placement'].mean():.3f} ===")
        print("Traits:  " + ", ".join(f"{t['name']} ({t['freq']:.0%}, x{t['lift']:.1f})" for t in traits))
        print("Board:   " + ", ".join(
            f"{c['name']} ({c['freq']:.0%}, {c['avg_star']:.1f}-star, x{c['lift']:.1f})" for c in champs
        ))


def save_cluster_assignments(df, labels, out_path=None):
    if out_path is None:
        out_path = set_suffixed_path("archetype_clusters", "csv")
    out = df.copy()
    out["archetype_cluster"] = labels
    out[["participant_id", "match_id", "seed_tier", "patch", "placement", "archetype_cluster"]].to_csv(
        out_path, index=False
    )
    return out_path


def main(k=20):
    print("Loading composition matrix (binarized trait/champion presence)")
    df, X, comp_cols = load_composition_matrix()
    print(f"  {len(X)} rows, {len(comp_cols)} composition columns "
          f"({sum(c.startswith('trait__') for c in comp_cols)} traits, "
          f"{sum(c.startswith('champ__') for c in comp_cols)} champions)")

    print(f"\nFitting final KMeans with k={k} on the full dataset")
    km, labels = fit_final(X, k)

    raw = df[comp_cols]

    print("\nDetailed cluster reports (traits/champs that are both common within the "
          "cluster and distinctive vs. the overall population):")
    print_cluster_reports(df, X, raw, labels)

    out_path = save_cluster_assignments(df, labels)
    print(f"\nSaved per-participant cluster assignment to {out_path}")


if __name__ == "__main__":
    main(k=20)
