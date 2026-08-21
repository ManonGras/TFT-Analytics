import pandas as pd

CLUSTERS_PATH = "data/processed/archetype_clusters.csv"


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

    out_path = "data/processed/archetype_contest.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()