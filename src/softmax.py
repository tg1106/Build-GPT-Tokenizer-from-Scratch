# Day 7: exp, log, softmax, cross-entropy loss and perplexity.
# Run from the project root:  python3 src/softmax.py

# math is part of Python itself (nothing to install).
import math

print("e =", round(math.e, 4))
print("exp(0), exp(1), exp(2):", [round(math.exp(v), 3) for v in [0, 1, 2]])  # list comprehension: a loop in one line
print("log(1), log(0.5), log(0.1):", [round(math.log(v), 3) for v in [1, 0.5, 0.1]])
assert abs(math.log(math.exp(2.5)) - 2.5) < 1e-12  # log undoes exp


# softmax: scores (logits) in, probabilities out.
def softmax(logits):
    # Subtracting the biggest score changes nothing (it cancels top and bottom)
    # but stops exp() overflowing on large scores.
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


probs = softmax([2, 1, 0])
print("\nsoftmax([2, 1, 0]):", [round(p, 3) for p in probs])
assert abs(sum(probs) - 1) < 1e-9
print("softmax([1000, 1001]):", [round(p, 3) for p in softmax([1000, 1001])])  # would crash without the max trick


# Cross-entropy loss: minus the log of the probability given to the correct answer.
def cross_entropy(probs, correct):
    return -math.log(probs[correct])


print("\nloss if correct is index 0:", round(cross_entropy(probs, 0), 3))  # likely answer: small loss
print("loss if correct is index 2:", round(cross_entropy(probs, 2), 3))  # unlikely answer: big loss

# Baseline: a model that guesses all 65 characters equally.
uniform = softmax([0] * 65)  # equal scores give equal probabilities, each 1/65
loss = cross_entropy(uniform, 0)
print("\nuniform guess over 65 chars: loss", round(loss, 3), "| perplexity", round(math.exp(loss), 1))
assert abs(math.exp(loss) - 65) < 1e-9  # perplexity = e^loss = number of equally likely choices
