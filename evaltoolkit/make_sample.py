"""Write a synthetic judge-vs-human CSV for the demo. No real data."""
from __future__ import annotations

import csv
import random

VERDICTS = ["send", "revise", "review"]


def make_rows(n=120, seed=11):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        human = rng.choices(VERDICTS, weights=[0.45, 0.35, 0.20])[0]
        # the judge agrees most of the time, and leans to "revise" when it disagrees
        judge = human if rng.random() < 0.72 else rng.choices(VERDICTS, weights=[0.15, 0.6, 0.25])[0]
        base = {"send": 7.5, "revise": 4.5, "review": 5.0}[human]
        rows.append({
            "item_id": f"draft-{i + 1:03d}",
            "judge_verdict": judge,
            "human_verdict": human,
            "score_reads_human": round(min(10, max(0, rng.gauss(base, 1.5))), 1),
            "score_grounded": round(min(10, max(0, rng.gauss(base + 0.5, 1.5))), 1),
            "score_tone_match": round(min(10, max(0, rng.gauss(base - 1.5, 1.8))), 1),
        })
    return rows


def main(path="data/sample_judgements.csv"):
    rows = make_rows()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
