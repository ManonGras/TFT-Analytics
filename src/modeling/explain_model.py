import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from modeling.train_model import load_and_prepare, train

SAMPLE_SIZE = 2000


def main():
    print("Loading data and training model")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_prepare()
    model = train(X_train, y_train)

    print(f"Computing SHAP values on a sample of {SAMPLE_SIZE} test rows")
    sample = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    mean_abs_shap = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns)
        .sort_values(ascending=False)
    )
    mean_signed_shap = pd.Series(shap_values.mean(axis=0), index=sample.columns)

    summary = pd.DataFrame({
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap.reindex(mean_abs_shap.index),
    })
    out_path = "data/processed/shap_feature_importance.csv"
    summary.to_csv(out_path)
    print(f"Saved full ranking (with signed mean) to {out_path}")

    print("\n--- Top 20 by mean |SHAP|: magnitude and direction ---")
    print("(negative mean_signed_shap = pushes toward a BETTER placement, since 1=best)")
    print(summary.head(20).to_string())

    print("\n--- seed_tier: mean SHAP contribution per tier (negative = pushes toward better placement) ---")
    print("(this is the per-tier breakdown of the pooled seed_tier effect shown above)")
    tier_idx = sample.columns.get_loc("seed_tier")
    tier_breakdown = (
        pd.DataFrame({
            "seed_tier": test_df.loc[sample.index, "seed_tier"].astype(str),
            "shap": shap_values[:, tier_idx],
        })
        .groupby("seed_tier")["shap"]
        .agg(mean="mean", n="count")
        .sort_values("mean")
    )
    print(tier_breakdown.to_string())

    print("\n--- patch: mean SHAP contribution per patch ---")
    print("(only 16.15/16.16 appear here - the test set only covers those two held-out patches)")
    patch_idx = sample.columns.get_loc("patch")
    patch_breakdown = (
        pd.DataFrame({
            "patch": test_df.loc[sample.index, "patch"].astype(str),
            "shap": shap_values[:, patch_idx],
        })
        .groupby("patch")["shap"]
        .agg(mean="mean", n="count")
        .sort_values("mean")
    )
    print(patch_breakdown.to_string())

    print("\nRendering SHAP beeswarm plot")
    plt.figure()
    shap.summary_plot(shap_values, sample, max_display=25, show=False)
    plot_path = "data/processed/shap_summary_plot.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
