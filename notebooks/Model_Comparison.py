from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("output")


files = [
    OUTPUT_DIR /
    "linear_regression_metrics.csv",

    OUTPUT_DIR /
    "random_forest_metrics.csv",

    OUTPUT_DIR /
    "arima_metrics.csv"
]


results = []

for file in files:

    if file.exists():

        results.append(
            pd.read_csv(file)
        )


if not results:

    raise FileNotFoundError(
        "No model metrics found. "
        "Run the models first."
    )


comparison = pd.concat(
    results,
    ignore_index=True
)


comparison = comparison.sort_values(
    by="MAE",
    ascending=True
)


print()
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    OUTPUT_DIR /
    "model_comparison.csv",
    index=False
)


print()
print(
    "Saved to:",
    OUTPUT_DIR /
    "model_comparison.csv"
)