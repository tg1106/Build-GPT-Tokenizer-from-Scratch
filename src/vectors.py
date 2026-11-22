# Day 3: vectors and the dot product with NumPy.
# Run from the project root:  venv_gpt/bin/python src/vectors.py

# import loads a library. NumPy is the standard library for arrays of numbers.
# "as np" gives it the short name np.
import numpy as np

# np.array turns a list into a NumPy array. A 1-D array is a vector.
a = np.array([3, 4])
print("a:", a, "| shape:", a.shape)  # shape (2,) means: 2 numbers in one row

# Length of the arrow: square each number, add, take the square root (Pythagoras).
# ** is "to the power of". a**2 squares every number at once, no loop needed.
length = np.sqrt(np.sum(a**2))
print("length of a:", length)  # 3*3 + 4*4 = 25, sqrt = 5
assert length == 5.0


# Dot product by hand: multiply position by position, then add everything up.
def dot_loop(x, y):
    total = 0
    for xi, yi in zip(x, y):  # zip pairs up x[0] with y[0], x[1] with y[1], ...
        total += xi * yi
    return total


x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("\ndot by loop:", dot_loop(x, y))  # 1*4 + 2*5 + 3*6 = 32
print("dot by np.dot:", np.dot(x, y))
assert dot_loop(x, y) == np.dot(x, y) == 32

# The dot product says how much two arrows point the same way.
right = np.array([1, 0])
print("\nsame direction:", np.dot(right, np.array([2, 0])))  # positive
print("perpendicular: ", np.dot(right, np.array([0, 1])))  # zero
print("opposite:      ", np.dot(right, np.array([-3, 0])))  # negative
