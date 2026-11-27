# Day 13: the bigram model as a trained model: one-hot, softmax, cross-entropy, gradient descent.
# Run from the project root:  venv_gpt/bin/python src/bigram_trained.py

import numpy as np

np.random.seed(1)  # fixed seed: the same mini-batches every run

with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}  # dict comprehension: a loop that builds a dict
V = len(chars)  # vocabulary size, 65

# ids[k] is the id of the k-th character; xs are the inputs, ys the characters that follow them.
ids = np.array([stoi[ch] for ch in text])
xs, ys = ids[:-1], ids[1:]


# A one-hot vector times W just picks one row of W (these two lines give the same answer).
W = np.random.randn(V, V)
x = np.zeros(V)
x[stoi["t"]] = 1  # one-hot for 't'
assert np.allclose(x @ W, W[stoi["t"]])  # allclose: equal up to tiny rounding


def softmax_rows(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))  # axis=1: work along each row
    return e / e.sum(axis=1, keepdims=True)


# Loss over all pairs: -mean log p(correct next char). Using the 65x65 table of
# probabilities avoids building a 1.1-million-row matrix.
def full_loss(W):
    logp = np.log(softmax_rows(W))
    return -logp[xs, ys].mean()  # logp[xs, ys] picks logp[xs[k], ys[k]] for every pair k


# Gradient of the mean loss for a batch: probs - one_hot(target), averaged, added to the row of each input.
def grad(W, bx, by):
    p = softmax_rows(W[bx])  # W[bx] is one row of W per input char: shape (batch, 65)
    p[np.arange(len(by)), by] -= 1  # subtract the one-hot target
    p /= len(by)
    dW = np.zeros_like(W)
    np.add.at(dW, bx, p)  # add.at sums correctly when the same row appears many times
    return dW


# Check the gradient by nudging one entry of W (as on day 5).
W0 = np.random.randn(V, V)
bx, by = xs[:200], ys[:200]


def batch_loss(W):
    return -np.log(softmax_rows(W[bx]))[np.arange(200), by].mean()


i, j, h = stoi["t"], stoi["h"], 1e-6
W1 = W0.copy()
W1[i, j] += h
nudged = (batch_loss(W1) - batch_loss(W0)) / h
print(f"gradient check at W['t','h']: formula {grad(W0, bx, by)[i, j]:.5f}, nudged {nudged:.5f}")
assert abs(grad(W0, bx, by)[i, j] - nudged) < 1e-4

# Train. W starts at zeros, so every row is uniform and the loss must be log(65).
W = np.zeros((V, V))
print(f"\nstart loss {full_loss(W):.4f}  (log 65 = {np.log(V):.4f})")
assert abs(full_loss(W) - np.log(V)) < 1e-12

lr, batch = 100.0, 16384
for step in range(1, 1001):
    pick = np.random.randint(0, len(xs), batch)  # a random batch of pair numbers
    W -= lr * grad(W, xs[pick], ys[pick])
    if step % 200 == 0:
        print(f"step {step:4}  loss {full_loss(W):.4f}")

# Compare with the counting model from day 12.
counts = np.zeros((V, V))
np.add.at(counts, (xs, ys), 1)  # counts[a, b] = how often b follows a
P_counts = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
loss_counts = -np.log(P_counts[xs, ys]).mean()
print(f"\ncounting model loss: {loss_counts:.4f}")
print(f"trained model loss:  {full_loss(W):.4f}")

row = stoi["q"]
print("after 'q', P('u'): counts", round(P_counts[row, stoi["u"]], 3), "| trained", round(softmax_rows(W)[row, stoi["u"]], 3))
assert full_loss(W) < loss_counts + 0.02  # the trained table cannot beat counting, only approach it
