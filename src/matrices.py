# Day 4: matrices, matrix multiply, and shapes.
# Run from the project root:  venv_gpt/bin/python src/matrices.py

import numpy as np

# A list of lists makes a 2-D array: a matrix. shape is (rows, columns).
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("A shape:", A.shape)


# Matrix multiply by hand: entry (i, j) = dot product of row i of A with column j of B.
def matmul_loop(X, Y):
    rows, inner = X.shape  # unpack the shape tuple into two names
    inner2, cols = Y.shape
    assert inner == inner2, f"inner sizes differ: {inner} vs {inner2}"
    out = np.zeros((rows, cols))  # empty grid of zeros to fill in
    for i in range(rows):  # range(3) gives 0, 1, 2
        for j in range(cols):
            # X[i] is row i. Y[:, j] is column j (: means "all rows").
            out[i, j] = np.dot(X[i], Y[:, j])
    return out


print("\nA @ B by loop:\n", matmul_loop(A, B))
# @ is NumPy's matrix multiply.
print("A @ B by @:\n", A @ B)
assert (matmul_loop(A, B) == A @ B).all()  # .all() is True only if every entry matches

# Shapes: (m, n) @ (n, p) -> (m, p).
C = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
D = np.array([[1, 0], [0, 1], [1, 1]])  # shape (3, 2)
print("\nC", C.shape, "@ D", D.shape, "->", (C @ D).shape)

# When inner sizes differ, NumPy refuses. try/except catches the error so the script goes on.
try:
    C @ C
except ValueError:
    print("C @ C fails: shapes (2, 3) and (2, 3), inner sizes 3 and 2 differ")

# .T transposes: rows become columns. This is the K^T in attention.
print("\nC.T shape:", C.T.shape)
# Pretend C holds 2 query vectors and 2 key vectors of length 3.
# Q @ K^T has shape (2, 2): entry (i, j) = how well query i matches key j.
print("C @ C.T:\n", C @ C.T)
