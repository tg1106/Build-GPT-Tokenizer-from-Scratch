# Day 12: a bigram model from counts: negative log likelihood and sampling.
# Run from the project root:  python3 src/bigram_model.py

import math
import random

random.seed(1)  # fixed seed so the sampled text is the same every run

with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()


# Row a of the table: for each character that follows a, its probability.
# shortcut: a pair never seen in s gets probability 0, so scoring other text with this can hit log(0);
# upgrade to smoothing (add 1 to every count) when we score held-out text.
def build_probs(s):
    counts = {}
    for a, b in zip(s, s[1:]):
        row = counts.setdefault(a, {})  # setdefault: get the row, or make an empty one
        row[b] = row.get(b, 0) + 1
    probs = {}
    for a, row in counts.items():
        total = sum(row.values())
        probs[a] = {b: n / total for b, n in row.items()}
    return probs


# Average negative log likelihood: the mean of -log p(next | prev) over all pairs.
def avg_nll(probs, s):
    total = 0.0
    for a, b in zip(s, s[1:]):
        total += -math.log(probs[a][b])
    return total / (len(s) - 1)


# Start from one character and repeatedly draw the next one from its row.
def sample(probs, start, n):
    out = start
    for _ in range(n):
        row = probs[out[-1]]  # out[-1] is the last character so far
        # random.choices picks from a list using weights; k=1 returns a list of one pick.
        out += random.choices(list(row.keys()), weights=list(row.values()), k=1)[0]
    return out


# Hand example: "aab" has pairs a->a and a->b, so each probability is 0.5.
tiny = build_probs("aab")
print("probs from 'aab':", tiny)
nll = avg_nll(tiny, "aab")
print(f"avg NLL {nll:.3f} (= log 2), perplexity {math.exp(nll):.2f}")
assert abs(nll - math.log(2)) < 1e-12

# The whole book.
probs = build_probs(text)
nll = avg_nll(probs, text)
print(f"\nbigram model on the book: avg NLL {nll:.3f}, perplexity {math.exp(nll):.1f}")
print(f"uniform guess over 65 chars: avg NLL {math.log(65):.3f}, perplexity 65.0")
assert nll < math.log(65)  # the table must beat blind guessing

print("\nsampled text:")
print(sample(probs, "\n", 200))
