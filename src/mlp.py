# Day 11: a neuron, a layer and a small MLP built from Value, trained by gradient descent.
# Run from the project root:  python3 src/mlp.py

import random

# Our own file value.py, from the same folder. Its demo does not run on import.
from value import Value

random.seed(1)  # fixed seed: the "random" starting weights are the same every run


class Neuron:
    def __init__(self, n_in):
        # one random weight per input, between -1 and 1, plus a bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]  # _ is a name for "not used"
        self.b = Value(0.0)

    # __call__ lets us use the object like a function: neuron(x)
    def __call__(self, x):
        # w1*x1 + w2*x2 + ... + bias, squashed by tanh.
        # sum() starts from 0, which is why Value needed __radd__.
        total = sum([wi * xi for wi, xi in zip(self.w, x)]) + self.b
        return total.tanh()

    def parameters(self):
        return self.w + [self.b]  # + joins two lists


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]  # every neuron reads the same inputs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, sizes):
        # MLP(3, [4, 4, 1]) makes layers 3->4, 4->4, 4->1
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:  # each layer's output feeds the next
            x = layer(x)
        return x[0]  # the last layer has one neuron, so return its single output

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# Toy data: 4 examples with 3 numbers each, and a target of +1 or -1 for each.
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])
params = model.parameters()
print("number of parameters:", len(params))  # (3*4+4) + (4*4+4) + (4*1+1) = 41


def total_loss():
    # sum of (prediction - target)^2 over all examples
    return sum([(model(x) - y) ** 2 for x, y in zip(xs, ys)])


lr = 0.05  # learning rate: the size of each step
first = total_loss().data
for step in range(21):
    loss = total_loss()
    for p in params:
        p.grad = 0.0  # reset: grads add up with +=, so clear the old ones first
    loss.backward()
    for p in params:
        p.data -= lr * p.grad  # step against the slope
    if step % 5 == 0:
        print(f"step {step:2}  loss {loss.data:.4f}")  # % is the remainder: true every 5th step

print("\npredictions:", [round(model(x).data, 3) for x in xs])
print("targets:    ", ys)
assert total_loss().data < first / 10  # the loss must have dropped a lot
