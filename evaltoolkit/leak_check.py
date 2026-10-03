"""Show how a random split inflates nearest-neighbour accuracy when near-duplicates share a group.

Photos from the same inspection (or messages from the same thread, pages from the
same document) are near-duplicates. If you split by item, each test item's siblings
sit in the training set and vote for it, so accuracy looks far better than it will
be on a new group. Split by group instead: that is how production meets new data.

    python -m evaltoolkit.leak_check

Runs on synthetic data only: groups of near-duplicate vectors with noisy labels.
"""
from __future__ import annotations

import math
import random
from collections import Counter


def make_dataset(n_groups=120, per_group=8, dims=16, n_classes=6, seed=7):
    """Each group has a random centre; its items are small perturbations of it.
    The class signal is weak on purpose, so only near-duplicates make it look easy."""
    rng = random.Random(seed)
    class_dirs = [[rng.gauss(0, 1) for _ in range(dims)] for _ in range(n_classes)]
    items = []
    for g in range(n_groups):
        centre = [rng.gauss(0, 1) for _ in range(dims)]
        group_label = rng.randrange(n_classes)
        for _ in range(per_group):
            # siblings mostly share a label, like photos of one section in one inspection
            label = group_label if rng.random() < 0.8 else rng.randrange(n_classes)
            vec = [c + 0.45 * d + rng.gauss(0, 0.15)
                   for c, d in zip(centre, class_dirs[label])]
            items.append({"group": g, "label": label, "vec": vec})
    return items


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def knn_predict(query, library, k=5):
    nearest = sorted(library, key=lambda it: -cosine(query["vec"], it["vec"]))[:k]
    return Counter(it["label"] for it in nearest).most_common(1)[0][0]


def accuracy(test, train, k=5):
    return sum(knn_predict(q, train, k) == q["label"] for q in test) / len(test)


def random_split(items, test_share=0.25, seed=1):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * test_share)
    return shuffled[:cut], shuffled[cut:]


def group_split(items, test_share=0.25, seed=1):
    rng = random.Random(seed)
    groups = sorted({it["group"] for it in items})
    rng.shuffle(groups)
    test_groups = set(groups[: int(len(groups) * test_share)])
    test = [it for it in items if it["group"] in test_groups]
    train = [it for it in items if it["group"] not in test_groups]
    return test, train


def main():
    items = make_dataset()
    test_r, train_r = random_split(items)
    test_g, train_g = group_split(items)
    leaky = accuracy(test_r, train_r)
    honest = accuracy(test_g, train_g)
    print(f"Items: {len(items)} in {len({i['group'] for i in items})} groups")
    print(f"Random split (leaky):    {leaky:.1%}")
    print(f"Group split (honest):    {honest:.1%}")
    print(f"Inflation from the leak: {100 * (leaky - honest):.1f} points")


if __name__ == "__main__":
    main()
