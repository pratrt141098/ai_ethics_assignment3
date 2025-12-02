import csv
import math
from collections import defaultdict

IN_FILE = "data/mixed_scores.csv"
OUT_FILE = "data/metrics_indicator_model_mixed.csv"


def mean_std(values):
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def main():
    scores = defaultdict(list)  # (indicator_id, model_name) -> list[mixed_score]

    with open(IN_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indicator_id = row["indicator_id"]
            model_name = row["model_name"]
            mixed_str = row["mixed_score"]
            if not mixed_str:
                continue
            mixed = float(mixed_str)
            scores[(indicator_id, model_name)].append(mixed)

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "indicator_id",
            "model_name",
            "n_items",
            "mean_mixed_score",
            "std_mixed_score",
        ])

        for (indicator_id, model_name), vals in sorted(scores.items()):
            m, s = mean_std(vals)
            writer.writerow([
                indicator_id,
                model_name,
                len(vals),
                f"{m:.3f}" if m is not None else "",
                f"{s:.3f}" if s is not None else "",
            ])


if __name__ == "__main__":
    main()
