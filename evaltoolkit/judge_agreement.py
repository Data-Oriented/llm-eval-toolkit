"""Check an LLM judge against human labels before you trust it.

Input: a CSV with one row per judged item and at least the columns
`judge_verdict` and `human_verdict` (for example send / revise / review).
Optional numeric columns are summarised as per-criterion score medians.

    python -m evaltoolkit.judge_agreement data/sample_judgements.csv

Reports raw agreement, Cohen's kappa (agreement corrected for chance),
a confusion matrix, per-verdict precision and recall, and the items where
the judge and the human disagree, which is where the rubric needs work.
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from dataclasses import dataclass


@dataclass
class AgreementReport:
    n: int
    labels: list[str]
    agreement: float
    kappa: float
    confusion: dict[tuple[str, str], int]
    precision: dict[str, float]
    recall: dict[str, float]


def cohen_kappa(judge: list[str], human: list[str]) -> float:
    """Cohen's kappa for two raters over the same items."""
    if len(judge) != len(human) or not judge:
        raise ValueError("need two non-empty label lists of equal length")
    n = len(judge)
    observed = sum(j == h for j, h in zip(judge, human)) / n
    jc, hc = Counter(judge), Counter(human)
    expected = sum(jc[k] * hc[k] for k in set(jc) | set(hc)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def agreement_report(judge: list[str], human: list[str]) -> AgreementReport:
    labels = sorted(set(judge) | set(human))
    confusion = Counter(zip(judge, human))
    precision, recall = {}, {}
    for lab in labels:
        tp = confusion[(lab, lab)]
        judged = sum(v for (j, _), v in confusion.items() if j == lab)
        actual = sum(v for (_, h), v in confusion.items() if h == lab)
        precision[lab] = tp / judged if judged else 0.0
        recall[lab] = tp / actual if actual else 0.0
    agreement = sum(j == h for j, h in zip(judge, human)) / len(judge)
    return AgreementReport(len(judge), labels, agreement, cohen_kappa(judge, human),
                           dict(confusion), precision, recall)


def load(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(path: str) -> None:
    rows = load(path)
    judge = [r["judge_verdict"].strip() for r in rows]
    human = [r["human_verdict"].strip() for r in rows]
    rep = agreement_report(judge, human)

    print(f"Items: {rep.n}")
    print(f"Raw agreement: {rep.agreement:.1%}   Cohen's kappa: {rep.kappa:.2f}")
    print("\nConfusion (rows = judge, columns = human):")
    print("judge\\human".ljust(14) + "".join(l.ljust(10) for l in rep.labels))
    for j in rep.labels:
        print(j.ljust(14) + "".join(str(rep.confusion.get((j, h), 0)).ljust(10) for h in rep.labels))
    print("\nPer verdict:")
    for lab in rep.labels:
        print(f"  {lab:<10} precision {rep.precision[lab]:.0%}   recall {rep.recall[lab]:.0%}")

    score_cols = [c for c in rows[0] if c.startswith("score_")]
    if score_cols:
        print("\nScore medians (and share under 5/10):")
        for c in score_cols:
            vals = [float(r[c]) for r in rows if r[c] != ""]
            under = sum(v < 5 for v in vals) / len(vals)
            print(f"  {c[6:]:<16} median {statistics.median(vals):.1f}   under 5: {under:.0%}")

    disagreements = [r for r, j, h in zip(rows, judge, human) if j != h]
    print(f"\nDisagreements to review: {len(disagreements)}")
    for r in disagreements[:10]:
        print(f"  {r.get('item_id', '?')}: judge={r['judge_verdict']} human={r['human_verdict']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_judgements.csv")
