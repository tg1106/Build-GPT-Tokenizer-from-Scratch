# Day 8-9: a Value is a number that remembers how it was made, and can pass gradients back.
# Run from the project root:  python3 src/value.py


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
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Slope of a*b with respect to a is b, and with respect to b is a.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


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

# Backward pass, in order from the output back: d first, then e.
d.grad = 1.0  # nudging d by 1 moves d by 1
d._backward()  # pushes grad to e and c
e._backward()  # pushes grad to a and b
print("\nafter backward:")
for name, v in [("a", a), ("b", b), ("c", c), ("e", e), ("d", d)]:
    print(f"  {name}.grad = {v.grad}")
assert (a.grad, b.grad, c.grad, e.grad) == (-3, 2, 1, 1)


# Check against nudging: rebuild d from raw numbers and measure the slope.
def build(av, bv):
    return (Value(av) * Value(bv) + Value(10)).data


h = 0.001
print("\nnudged slope in a:", round((build(2 + h, -3) - build(2, -3)) / h, 3))
print("nudged slope in b:", round((build(2, -3 + h) - build(2, -3)) / h, 3))
assert abs((build(2 + h, -3) - build(2, -3)) / h - a.grad) < 0.01
assert abs((build(2, -3 + h) - build(2, -3)) / h - b.grad) < 0.01

# A value used twice: s = x + x, so ds/dx = 2. The two shares add up.
x = Value(5)
s = x + x
s.grad = 1.0
s._backward()
print("\nx used twice, x.grad =", x.grad)
assert x.grad == 2
