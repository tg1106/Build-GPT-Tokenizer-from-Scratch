# Day 2: count which character follows which (bigram counts), then turn counts into probabilities.
# Run from the project root:  python3 src/bigrams.py

with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()


def count_pairs(s):
    counts = {}
    # s[1:] is a slice: the string without its first character.
    # zip(s, s[1:]) walks both side by side, giving neighbour pairs: ('h','e'), ('e','l'), ...
    # Each pair is a tuple: a fixed group of values. Tuples can be dict keys.
    for pair in zip(s, s[1:]):
        # counts.get(pair, 0) returns the stored count, or 0 if the pair is new.
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def next_probs(counts, ch):
    # Total number of times anything followed ch.
    total = 0
    # .items() gives (key, value) pairs; here key is the tuple (a, b) and value is the count n.
    for (a, b), n in counts.items():
        if a == ch:
            total += n  # += means "add to the variable"
    # Probability of b after ch = its count divided by the total.
    probs = {}
    for (a, b), n in counts.items():
        if a == ch:
            probs[b] = n / total
    return probs


# A function used only to tell sorted() what to sort by: the count, item[1].
def by_count(item):
    return item[1]


# Check against the hand example.
small = count_pairs("hello")
print("hello pairs:", small)
assert small == {("h", "e"): 1, ("e", "l"): 1, ("l", "l"): 1, ("l", "o"): 1}
print("after 'l' in hello:", next_probs(small, "l"))

# Now the whole book.
counts = count_pairs(text)
print(f"\ndistinct pairs in the book: {len(counts)}")

# reverse=True sorts biggest first; [:5] keeps the first five.
print("most common pairs:")
for (a, b), n in sorted(counts.items(), key=by_count, reverse=True)[:5]:
    print(f"  {repr(a + b):8} {n}")

for ch in ["t", "q"]:
    probs = next_probs(counts, ch)
    # Every row of probabilities must add up to 1 (allowing tiny rounding error).
    assert abs(sum(probs.values()) - 1) < 1e-9
    print(f"\nmost likely characters after {repr(ch)}:")
    for b, p in sorted(probs.items(), key=by_count, reverse=True)[:5]:
        print(f"  {repr(b):6} {p:.3f}")
