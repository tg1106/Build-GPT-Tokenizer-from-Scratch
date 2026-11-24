# Day 6: partial derivatives and the chain rule, worked by hand and checked by nudging.
# Run from the project root:  python3 src/chain_rule.py


# Same nudge-and-measure slope as derivatives.py.
def slope(f, x, h=0.001):
    return (f(x + h) - f(x)) / h


# Partial derivatives of f(x, y) = x^2 * y.
def f(x, y):
    return x**2 * y


# To nudge only x, freeze y at 2. These helpers take one input each.
def f_x(x):
    return f(x, 2)


def f_y(y):
    return f(3, y)


# By hand: df/dx = 2xy = 2*3*2 = 12.  df/dy = x^2 = 9.
print("df/dx at (3,2): by hand 12, nudged", round(slope(f_x, 3), 3))
print("df/dy at (3,2): by hand  9, nudged", round(slope(f_y, 2), 3))
assert abs(slope(f_x, 3) - 12) < 0.01
assert abs(slope(f_y, 2) - 9) < 0.01


# Chain rule on y = (3x + 1)^2, split in two steps: u = 3x + 1, then y = u^2.
def step_u(x):
    return 3 * x + 1


def step_y(u):
    return u**2


def whole(x):
    return step_y(step_u(x))  # feed the first step's output into the second


x = 2
u = step_u(x)  # 7
du_dx = 3  # slope of u = 3x + 1
dy_du = 2 * u  # slope of y = u^2 is 2u = 14
by_hand = dy_du * du_dx  # chain rule: slopes multiply
print(f"\ndy/dx at x=2: by hand {dy_du} * {du_dx} = {by_hand}, nudged", round(slope(whole, x), 3))
assert abs(slope(whole, x) - by_hand) < 0.02
