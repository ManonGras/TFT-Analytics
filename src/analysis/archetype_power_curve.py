import pandas as pd

from db import get_engine
from set_paths import set_suffixed_path

CLUSTERS_PATH = set_suffixed_path("archetype_clusters", "csv")


def load_power_curve_data(engine):
    clusters = pd.read_csv(CLUSTERS_PATH)
    timing = pd.read_sql(
        "SELECT id AS participant_id, last_round, time_eliminated, level FROM participants",
        engine,
    )
    return clusters.merge(timing, on="participant_id", how="left")


def summarize_power_curve(df):
    summary = df.groupby("archetype_cluster").agg(
        n=("participant_id", "count"),
        avg_placement=("placement", "mean"),
        avg_last_round=("last_round", "mean"),
        median_last_round=("last_round", "median"),
        avg_time_eliminated=("time_eliminated", "mean"),
    ).round(2)
    return summary.sort_values("avg_last_round")


def summarize_metric_by_placement(df, metric_col, min_n=100):
    """bad boards just die/stall early in TFT, so a raw table would
    re-states placement quality, not playstyle. This controls for it:
    within each fixed placement value, compare metric_col across
    archetypes, to isolate which archetypes reach a given outcome via
    a different path (typically lower level = faster/reroll) rather than
    just "better/worse"."""
    overall_by_placement = df.groupby("placement")[metric_col].mean()

    per_cell = df.groupby(["archetype_cluster", "placement"]).agg(
        n=("participant_id", "count"),
        avg_metric=(metric_col, "mean"),
    ).reset_index()
    per_cell["baseline"] = per_cell["placement"].map(overall_by_placement)
    per_cell["delta_vs_baseline"] = per_cell["avg_metric"] - per_cell["baseline"]

    reliable = per_cell[per_cell["n"] >= min_n]
    archetype_avg_delta = (
        reliable.groupby("archetype_cluster")
        .apply(lambda g: (g["delta_vs_baseline"] * g["n"]).sum() / g["n"].sum(), include_groups=False)
        .sort_values()
    )
    return archetype_avg_delta


def main():
    engine = get_engine()

    print("Loading power curve data")
    df = load_power_curve_data(engine)
    print(f"{len(df)} participants loaded")

    summary = summarize_power_curve(df)

    print("\nPower curve by archetype:")
    print(summary.to_string())

    out_path = set_suffixed_path("archetype_power_curve", "csv")
    summary.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    print("\nLast round by placement:")
    print("Negative = earlier elimination, positive = longer survival")
    round_delta = summarize_metric_by_placement(df, "last_round")
    print(round_delta.round(2).to_string())

    print("\nLevel by placement:")
    print("Negative = lower level, positive = higher level")
    level_delta = summarize_metric_by_placement(df, "level")
    print(level_delta.round(2).to_string())

if __name__ == "__main__":
    main()
