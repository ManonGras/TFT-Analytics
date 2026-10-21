import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error

from set_paths import set_suffixed_path
from modeling.train_model import clip_placement, load_and_prepare, train

SAMPLE_SIZE = 2000

# Seeds for the stability check of the per-tier SHAP breakdown: each seed
# changes the model's internal validation split (early stopping) and the SHAP
# sample. A per-tier effect that flips sign between seeds is not a finding.
STABILITY_SEEDS = [0, 1, 2, 3, 4]


def compute_shap_values(model, X_test, sample_size=SAMPLE_SIZE, random_state=42):
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=random_state)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return sample, shap_values


def shap_summary_table(sample, shap_values):
    mean_abs_shap = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns)
        .sort_values(ascending=False)
    )
    mean_signed_shap = pd.Series(shap_values.mean(axis=0), index=sample.columns)
    return pd.DataFrame({
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap.reindex(mean_abs_shap.index),
    })


def category_breakdown(sample, shap_values, test_df, column):
    col_idx = sample.columns.get_loc(column)
    return (
        pd.DataFrame({
            column: test_df.loc[sample.index, column].astype(str),
            "shap": shap_values[:, col_idx],
        })
        .groupby(column)["shap"]
        .agg(mean="mean", n="count")
        .sort_values("mean")
    )


def mae_without_column(X_train, y_train, X_test, y_test, column, random_state=42):
    """Test MAE with and without one feature: how much the model actually
    relies on it, independently of how SHAP spreads attributions."""
    with_col = train(X_train, y_train, random_state=random_state).predict(X_test)
    without = train(X_train.drop(columns=column), y_train, random_state=random_state).predict(
        X_test.drop(columns=column)
    )
    return {
        "mae_with": mean_absolute_error(y_test, clip_placement(with_col)),
        "mae_without": mean_absolute_error(y_test, clip_placement(without)),
    }


def breakdown_across_seeds(X_train, y_train, X_test, test_df, column, seeds=STABILITY_SEEDS,
                           sample_size=SAMPLE_SIZE):
    """Per-category mean SHAP of `column` for several seeds. Returns one row
    per category with the mean over seeds, the spread between seeds, and
    whether the sign is the same for every seed."""
    per_seed = {}
    for seed in seeds:
        model = train(X_train, y_train, random_state=seed)
        sample, shap_values = compute_shap_values(model, X_test, sample_size, random_state=seed)
        per_seed[seed] = category_breakdown(sample, shap_values, test_df, column)["mean"]
    table = pd.DataFrame(per_seed)
    return pd.DataFrame({
        "mean_over_seeds": table.mean(axis=1),
        "std_between_seeds": table.std(axis=1),
        "min": table.min(axis=1),
        "max": table.max(axis=1),
        "same_sign_all_seeds": (np.sign(table).nunique(axis=1) == 1),
    })


def main():
    print("Loading data and training model")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_prepare()
    model = train(X_train, y_train)

    print(f"Computing SHAP values on a sample of {SAMPLE_SIZE} test rows")
    sample, shap_values = compute_shap_values(model, X_test)

    summary = shap_summary_table(sample, shap_values)
    out_path = set_suffixed_path("shap_feature_importance", "csv")
    summary.to_csv(out_path)
    print(f"Saved full ranking (with signed mean) to {out_path}")

    print("\n--- Top 20 by mean |SHAP|: magnitude and direction ---")
    print("(negative mean_signed_shap = pushes toward a BETTER placement, since 1=best)")
    print(summary.head(20).to_string())

    print("\n--- seed_tier: mean SHAP contribution per tier (negative = pushes toward better placement) ---")
    print("(this is the per-tier breakdown of the pooled seed_tier effect shown above)")
    print(category_breakdown(sample, shap_values, test_df, "seed_tier").to_string())

    print("\n--- patch: mean SHAP contribution per patch ---")
    print("(the test set includes only the most recent matches, as determined by TEST_FRACTION in "
            "train_model.py.")
    print(category_breakdown(sample, shap_values, test_df, "patch").to_string())

    print("\n--- seed_tier: does the model need it? ---")
    ablation = mae_without_column(X_train, y_train, X_test, y_test, "seed_tier")
    print(f"  Test MAE with seed_tier: {ablation['mae_with']:.4f} | without: {ablation['mae_without']:.4f}")

    print(f"\n--- seed_tier: per-tier mean SHAP over {len(STABILITY_SEEDS)} seeds ---")
    stability = breakdown_across_seeds(X_train, y_train, X_test, test_df, "seed_tier")
    print(stability.round(4).to_string())
    out_path = set_suffixed_path("seed_tier_shap_stability", "csv")
    stability.to_csv(out_path)
    print(f"Saved to {out_path}")

    print("\nRendering SHAP beeswarm plot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    shap.summary_plot(shap_values, sample, max_display=25, show=False)
    plot_path = set_suffixed_path("shap_summary_plot", "png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
