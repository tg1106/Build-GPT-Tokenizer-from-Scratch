# Day 8-10: a Value is a number that remembers how it was made, and can pass gradients back.
# Run from the project root:  python3 src/value.py

import math


# A function that does nothing: plain inputs have nothing to pass back.
def do_nothing():
    pass


# A class is a blueprint. Each Value made from it is an object with its own data.
class Value:
    # __init__ runs when we write Value(...). self is the object being built.
    def __init__(self, data, children=(), op=""):
        self.data = data  # the number
        self.grad = 0.0  # how much the final output changes per unit nudge of this value
        self._backward = do_nothing  # a function that pushes grad to the children
        self._prev = children  # the Values this one was made from
        self._op = op  # the operation that made it ("" for a plain input)

    # __repr__ decides how a Value prints.
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # a + b calls a.__add__(b). The result remembers both inputs and the "+".
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # let plain numbers in
        out = Value(self.data + other.data, (self, other), "+")

        # A function defined inside a method remembers self, other and out (a closure).
        def _backward():
            # Chain rule: local slope (1 for add) * the grad of the result.
            # += because a value used twice gets a share from each use.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    # a * b calls a.__mul__(b).
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Slope of a*b with respect to a is b, and with respect to b is a.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # -a, a - b, and number + a, number * a (the "r" versions run when the number is on the left).
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # a ** n calls a.__pow__(n). n is a plain number, not a Value.
    def __pow__(self, n):
        out = Value(self.data**n, (self,), f"**{n}")

        def _backward():
            self.grad += n * self.data ** (n - 1) * out.grad  # slope of x^n is n * x^(n-1)

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad  # slope of e^x is e^x, which is out itself

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad  # slope of tanh is 1 - tanh^2

        out._backward = _backward
        return out

    # Run the whole backward pass from this value.
    def backward(self):
        order = []  # will hold the values with inputs first and this value last
        seen = set()  # a set remembers what we visited, so nothing is visited twice

        # Recursion: visit calls itself on each child before adding the value itself.
        def visit(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    visit(child)
                order.append(v)

        visit(self)
        self.grad = 1.0  # nudging the output by 1 moves the output by 1
        for v in reversed(order):  # output first, inputs last
            v._backward()


# The demo below runs only when this file is run directly, not when another file imports Value.
if __name__ == "__main__":
    a = Value(2)
    b = Value(-3)
    c = Value(10)
    e = a * b  # -6, made from (a, b) by *
    d = e + c  # 4, made from (e, c) by +
    print("e:", e, "| op:", e._op, "| made from:", e._prev)
    print("d:", d, "| op:", d._op, "| made from:", d._prev)

    assert e.data == -6 and d.data == 4
    assert d._prev == (e, c) and d._op == "+"  # tuples compare item by item
    assert e._prev == (a, b) and e._op == "*"
    assert a._prev == () and a._op == ""  # plain inputs have no history

    # Backward pass: backward() puts the values in order and runs every _backward for us.
    d.backward()
    print("\nafter backward:")
    for name, v in [("a", a), ("b", b), ("c", c), ("e", e), ("d", d)]:
        print(f"  {name}.grad = {v.grad}")
    assert (a.grad, b.grad, c.grad, e.grad) == (-3, 2, 1, 1)

    # A value used twice: s = x + x, so ds/dx = 2. The two shares add up.
    x = Value(5)
    s = x + x
    s.backward()
    print("\nx used twice, x.grad =", x.grad)
    assert x.grad == 2

    # A tiny neuron: o = tanh(x*w + bias). By hand the slope of tanh at 0.5 is 0.786.
    x = Value(0.5)
    w = Value(2)
    bias = Value(-0.5)
    o = (x * w + bias).tanh()
    o.backward()
    print(f"\nneuron: o = {o.data:.4f}")
    print(f"  x.grad = {x.grad:.4f}  w.grad = {w.grad:.4f}  bias.grad = {bias.grad:.4f}")
    assert abs(o.data - 0.4621) < 1e-4
    assert abs(x.grad - 1.5729) < 1e-4 and abs(w.grad - 0.3932) < 1e-4 and abs(bias.grad - 0.7864) < 1e-4


    # Check backward() against nudging, on a function where x feeds several paths.
    def f(v):
        return (v**2 + v.exp()).tanh()


    x0 = 0.3
    x = Value(x0)
    out = f(x)
    out.backward()
    h = 0.0001
    nudged = (f(Value(x0 + h)).data - f(Value(x0)).data) / h
    print(f"\ndf/dx at {x0}: backward {x.grad:.5f}, nudged {nudged:.5f}")
    assert abs(x.grad - nudged) < 1e-3
