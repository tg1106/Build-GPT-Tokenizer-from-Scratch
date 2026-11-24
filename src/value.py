# Day 8: a Value is a number that remembers how it was made (forward pass only).
# Run from the project root:  python3 src/value.py


# A class is a blueprint. Each Value made from it is an object with its own data.
class Value:
    # __init__ runs when we write Value(...). self is the object being built.
    def __init__(self, data, children=(), op=""):
        self.data = data  # the number
        self._prev = children  # the Values this one was made from
        self._op = op  # the operation that made it ("" for a plain input)

    # __repr__ decides how a Value prints.
    def __repr__(self):
        return f"Value(data={self.data})"

    # a + b calls a.__add__(b). The result remembers both inputs and the "+".
    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    # a * b calls a.__mul__(b).
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")


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
