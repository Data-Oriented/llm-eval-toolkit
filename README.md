# llm-eval-toolkit

Two small checks I run before trusting an AI system's numbers in production. Plain Python 3.10+, no dependencies, synthetic data only.

## 1. Is the LLM judge right? (`judge_agreement.py`)

An LLM judge that scores drafts is only useful if someone checks the judge. Label a sample of real outputs yourself, without looking at the judge's verdicts, then compare:

```bash
python -m evaltoolkit.make_sample                      # writes data/sample_judgements.csv (synthetic)
python -m evaltoolkit.judge_agreement data/sample_judgements.csv
```

It reports raw agreement, Cohen's kappa (agreement corrected for chance), a confusion matrix, precision and recall per verdict, per-criterion score medians, and the list of disagreements. The disagreements are where the rubric or the drafting prompt needs work.

What we learned using this on a production sales assistant:
- Label real production runs, not invented examples.
- Write down what "good" means before labelling.
- Low scores on every axis usually point at the drafting prompt, not at a harsh judge.
- Re-check whenever the prompt or the model changes.

## 2. Is the accuracy number inflated? (`leak_check.py`)

When items come in groups of near-duplicates (photos from one inspection, messages from one thread, pages from one document), a random split puts each test item's siblings in the training set, and a nearest-neighbour model looks far better than it will be on new data.

```bash
python -m evaltoolkit.leak_check
```

On the synthetic data, a random split scores about 84% and a split by group about 48%. In production we saw the same pattern: a photo classifier that reported 84% measured about 60% once photos from the same inspection could no longer vote for each other.

Rule: split evaluation data the way the system meets new data in production, and compare methods only on the same examples.

## Tests

```bash
python -m unittest -v
```

## Licence

MIT
