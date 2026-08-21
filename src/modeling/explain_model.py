import numpy as np
import pandas as pd
import shap

from modeling.train_model import load_and_prepare, train

SAMPLE_SIZE = 2000


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


def main():
    print("Loading data and training model")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_prepare()
    model = train(X_train, y_train)

    print(f"Computing SHAP values on a sample of {SAMPLE_SIZE} test rows")
    sample, shap_values = compute_shap_values(model, X_test)

    summary = shap_summary_table(sample, shap_values)
    out_path = "data/processed/shap_feature_importance.csv"
    summary.to_csv(out_path)
    print(f"Saved full ranking (with signed mean) to {out_path}")

    print("\n--- Top 20 by mean |SHAP|: magnitude and direction ---")
    print("(negative mean_signed_shap = pushes toward a BETTER placement, since 1=best)")
    print(summary.head(20).to_string())

    print("\n--- seed_tier: mean SHAP contribution per tier (negative = pushes toward better placement) ---")
    print("(this is the per-tier breakdown of the pooled seed_tier effect shown above)")
    print(category_breakdown(sample, shap_values, test_df, "seed_tier").to_string())

    print("\n--- patch: mean SHAP contribution per patch ---")
    print("(only 16.15/16.16 appear here - the test set only covers those two held-out patches)")
    print(category_breakdown(sample, shap_values, test_df, "patch").to_string())

    print("\nRendering SHAP beeswarm plot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    shap.summary_plot(shap_values, sample, max_display=25, show=False)
    plot_path = "data/processed/shap_summary_plot.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
