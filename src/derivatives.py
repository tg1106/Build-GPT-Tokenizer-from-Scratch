# Day 5: the derivative as a slope, found numerically by nudging the input.
# Run from the project root:  python3 src/derivatives.py


def square(x):
    return x**2


def cube(x):
    return x**3


# A function can be passed to another function like any value.
# slope(f, x) nudges x by h and measures how much f(x) moves per unit of nudge.
def slope(f, x, h=0.001):  # h=0.001 is a default: used when the caller gives no h
    return (f(x + h) - f(x)) / h


print("slope of x^2 at x=3:", slope(square, 3))  # exact answer 2x = 6
print("slope of x^3 at x=2:", slope(cube, 2))  # exact answer 3x^2 = 12
assert abs(slope(square, 3) - 6) < 0.01
assert abs(slope(cube, 2) - 12) < 0.02

# A smaller nudge h gives an answer closer to the exact 6.
print("\nslope of x^2 at x=3 as h shrinks:")
for h in [1, 0.1, 0.01, 0.001, 0.0001]:
    print(f"  h={h:<7} slope={slope(square, 3, h):.5f}")


# The sign of the slope tells which way is downhill.
# This bowl has its lowest point at x=3.
def bowl(x):
    return (x - 3) ** 2


print("\nslope of the bowl (x-3)^2:")
for x in [1, 3, 5]:
    print(f"  x={x}  slope={slope(bowl, x):+.3f}")  # + forces a sign to be printed
