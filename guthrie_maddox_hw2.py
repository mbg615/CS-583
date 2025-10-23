import bisect
import random
import sympy as sp
import numpy as np
from collections import Counter


# Helper Functions
def calculate_determinant_3x3(A: list[list[float]]) -> float:
    x1 = A[0][0] * A[1][1] * A[2][2]
    x2 = A[0][1] * A[1][2] * A[2][0]
    x3 = A[0][2] * A[1][0] * A[2][1]

    y1 = A[0][2] * A[1][1] * A[2][0]
    y2 = A[0][0] * A[1][2] * A[2][1]
    y3 = A[0][1] * A[1][0] * A[2][2]

    return x1 + x2 + x3 - y1 - y2 - y3


def calculate_determinant_4x4(A: list[list[float]]) -> float:
    x1 = A[0][0] * calculate_determinant_3x3([[A[1][1], A[1][2], A[1][3]],[A[2][1], A[2][2], A[2][3]],[A[3][1], A[3][2], A[3][3]]])
    x2 = A[0][1] * calculate_determinant_3x3([[A[1][0], A[1][2], A[1][3]],[A[2][0], A[2][2], A[2][3]],[A[3][0], A[3][2], A[3][3]]])
    x3 = A[0][2] * calculate_determinant_3x3([[A[1][0], A[1][1], A[1][3]],[A[2][0], A[2][1], A[2][3]],[A[3][0], A[3][1], A[3][3]]])
    x4 = A[0][3] * calculate_determinant_3x3([[A[1][0], A[1][1], A[1][2]],[A[2][0], A[2][1], A[2][2]],[A[3][0], A[3][1], A[3][2]]])

    return x1 - x2 + x3 - x4


def calculate_cofactor_4x4(A: list[list[float]]) -> list[list[float]]:
    C = [[0.0]*4 for _ in range(4)]

    for r in range(4):
        for c in range(4):
            minor = [[A[i][j] for j in range(4) if j != c] for i in range(4) if i != r]
            sign = (-1) ** (r+c)
            C[r][c] = sign * calculate_determinant_3x3(minor)

    return C


def calculate_transpose_4x4(A: list[list[float]]) -> list[list[float]]:
    T = [[A[j][i] for j in range(4)] for i in range(4)]
    return T


def multiply_4x4_by_scalar(A: list[list[float]], S) -> list[list[float]]:
    SA = [[S * A[i][j] for j in range(4)] for i in range(4)]
    return SA


def biased_toss() -> int:
    return 1 if random.random() < 0.7 else 0


# Q1
def fair_toss() -> int:
    toss1, toss2 = biased_toss(), biased_toss()

    if toss1 == 0 and toss2 == 1:
        return 0
    elif toss1 == 1 and toss2 == 0:
        return 1
    else:
        return fair_toss()


# Q2
def pick_index(w: list[int]) -> int:
    prefix = []
    total = 0
    for weight in w:
        total += weight
        prefix.append(total)
    target = random.uniform(0, total)
    return bisect.bisect_left(prefix, target)


# Q3
def reservoir_sample(infinite_stream, k: int) -> list:
    if k <= 0:
        return []

    reservoir = []
    for i in range(k):
        try:
            reservoir.append(next(infinite_stream))

        except StopIteration:
            return reservoir

    i = k
    while True:
        try:
            current_item = next(infinite_stream)
            i += 1

            j = random.randrange(i)

            if j < k:
                reservoir[j] = current_item

        except StopIteration:
            break

    return reservoir


# Q4
def subtract_lambda_identity_4x4(matrix, λ):
    return [[matrix[i][j] - (λ if i == j else 0) for j in range(4)] for i in range(4)]


def evaluate_characteristic_equation(coeffs, x):
    return coeffs[0]*x**4 + coeffs[1]*x**3 + coeffs[2]*x**2 + coeffs[3]*x + coeffs[4]


def differentiate_characteristic_equation(coeffs, x):
    return 4*coeffs[0]*x**3 + 3*coeffs[1]*x**2 + 2*coeffs[2]*x + coeffs[3]


def solve_quartic(coeffs):
    roots = []
    def newton_root(guess):
        x = guess
        for _ in range(1000):
            fx = evaluate_characteristic_equation(coeffs, x)
            dfx = differentiate_characteristic_equation(coeffs, x)
            if abs(dfx) < 1e-12:
                break
            x_new = x - fx / dfx
            if abs(x_new - x) < 1e-9:
                return x_new
            x = x_new
        return x
    for guess in [0, 1, -1, 2, -2, 5, -5, 10, -10]:
        root = newton_root(guess)
        if not any(abs(root - r) < 1e-6 for r in roots):
            roots.append(root)
        if len(roots) == 4:
            break
    return roots


def gaussian_solve(A, b):
    n = len(A)
    for i in range(n):
        pivot = A[i][i]
        if abs(pivot) < 1e-12:
            for k in range(i+1, n):
                if abs(A[k][i]) > 1e-12:
                    A[i], A[k] = A[k], A[i]
                    b[i], b[k] = b[k], b[i]
                    pivot = A[i][i]
                    break
        if abs(pivot) < 1e-12:
            continue
        for j in range(i, n):
            A[i][j] /= pivot
        b[i] /= pivot
        for k in range(i+1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]
    x = [0]*n
    for i in range(n-1, -1, -1):
        x[i] = b[i] - sum(A[i][j]*x[j] for j in range(i+1, n))
    return x


def calculate_eigenvalues_4x4(matrix: list[list[float]]) -> list[float]:
    xs = [-3, -1, 0, 1, 6]
    ys = [calculate_determinant_4x4(subtract_lambda_identity_4x4(matrix, x)) for x in xs]
    M = [[x**4, x**3, x**2, x, 1] for x in xs]
    coefficients = gaussian_solve(M, ys)
    roots = solve_quartic(coefficients)
    roots = [round(r) for r in sorted(roots, reverse=True)]
    return roots


# Q5
def calculate_inverse_4x4(matrix: list[list[float]]) -> list[list[float]] | None:
    det = calculate_determinant_4x4(matrix)
    if det == 0:
        return None

    cofactor_matrix = calculate_cofactor_4x4(matrix)
    adjugate = calculate_transpose_4x4(cofactor_matrix)
    matrix_inverse = multiply_4x4_by_scalar(adjugate, 1/det)

    return matrix_inverse


# Q6
def jacobi_svd_4x4(matrix: list[list[float]]) -> list[list[list[float]]]:
    A = sp.Matrix(matrix).evalf()
    ATA = (A.T * A).evalf()
    V = sp.eye(4)

    while True:
        max_val = 0
        p, q = 0, 0
        for i in range(4):
            for j in range(i + 1, 4):
                if abs(ATA[i, j]) > max_val:
                    max_val = abs(ATA[i, j])
                    p, q = i, j
        if max_val < 1e-12:
            break

        if ATA[p, q] == 0:
            continue

        tau = (ATA[q, q] - ATA[p, p]) / (2 * ATA[p, q])
        t = sp.sign(tau) / (abs(tau) + sp.sqrt(1 + tau**2))
        c = 1 / sp.sqrt(1 + t**2)
        s = c * t

        J = sp.eye(4)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = s
        J[q, p] = -s

        ATA = J.T * ATA * J
        V = V * J

    singular_vals = [sp.sqrt(max(ATA[i, i], 0)) for i in range(4)]
    idx = sorted(range(4), key=lambda i: -singular_vals[i])
    singular_vals = [singular_vals[i] for i in idx]
    V = V[:, idx]
    U = A * V
    for j in range(4):
        norm = sp.sqrt(sum(U[i, j]**2 for i in range(4)))
        if norm != 0:
            U[:, j] /= norm

    S = sp.zeros(4)
    for i in range(4):
        S[i, i] = singular_vals[i]

    U = U.evalf(6)
    S = S.evalf(6)
    V = V.evalf(6)

    return [U.tolist(), S.tolist(), V.T.tolist()]


# --- Function Demonstrations ---
# The following demonstration cases were generated by Google Gemini.

if __name__ == '__main__':
    # --- Q1: Fair Toss Demonstration ---
    # This test verifies that the `fair_toss` function produces a roughly
    # 50/50 distribution over many trials, despite using a biased source.
    print("--- Q1: Fair Toss Demonstration ---")
    toss_results = [fair_toss() for _ in range(1000)]
    counts = Counter(toss_results)
    print(f"Results of 1000 fair tosses (should be ~50/50): {counts}")
    print("-" * 40 + "\n")

    # --- Q2: Weighted Random Index Picker Demonstration ---
    # This test verifies that the `pick_index` function selects indices
    # according to their specified weights over a large number of trials.
    print("--- Q2: Weighted Random Index Picker Demonstration ---")
    weights = [1, 3, 10, 2]  # Probabilities: ~6%, ~19%, ~62%, ~13%
    num_picks = 10000
    picks = [pick_index(weights) for _ in range(num_picks)]
    pick_counts = Counter(picks)
    print(f"Weights: {weights}")
    print(f"Picks after {num_picks} trials:")
    for i in sorted(pick_counts.keys()):
        print(f"  Index {i}: {pick_counts[i]} times (~{(pick_counts[i] / num_picks) * 100:.2f}%)")
    print("-" * 40 + "\n")

    # --- Q3: Reservoir Sampling Demonstration ---
    # This shows the reservoir sampler selecting a small, fixed-size sample
    # from a much larger data stream.
    print("--- Q3: Reservoir Sampling Demonstration ---")
    data_stream = iter(range(1000))
    k = 10
    sample = reservoir_sample(data_stream, k)
    print(f"Sample of size k={k} from a stream of 1000 items: {sample}")

    # This demonstrates the case where k is larger than the stream size.
    data_stream_short = iter(range(5))
    k_large = 8
    sample_short = reservoir_sample(data_stream_short, k_large)
    print(f"Sample with k={k_large} from a stream of 5 items (returns all items): {sample_short}")
    print("-" * 40 + "\n")

    # --- Q4: Eigenvalue Calculation Demonstration ---
    # This calculates the eigenvalues for a known matrix. For a block
    # diagonal matrix, the eigenvalues are the eigenvalues of the blocks.
    print("--- Q4: Eigenvalue Calculation Demonstration ---")
    matrix_eig = [[4, 0, 1, 0], [0, -2, 0, 0], [1, 0, 4, 0], [0, 0, 0, 5]]
    print("Matrix:")
    for row in matrix_eig: print(f"  {row}")
    eigenvalues = calculate_eigenvalues_4x4(matrix_eig)
    print(f"\nCalculated Eigenvalues (sorted): {eigenvalues}")
    print("-" * 40 + "\n")

    # --- Q5: Matrix Inverse Demonstration ---
    # This demonstrates calculating the inverse of a matrix and also shows
    # that the function correctly identifies and handles a non-invertible
    # (singular) matrix.
    print("--- Q5: Matrix Inverse Demonstration ---")
    matrix_inv = [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]]
    print("Original Matrix:")
    for row in matrix_inv: print(f"  {row}")
    inverse = calculate_inverse_4x4(matrix_inv)
    print("\nInverse Matrix:")
    if inverse:
        for row in inverse: print(f"  {[f'{x:8.3f}' for x in row]}")
    else:
        print("  None (Matrix is singular)")

    print("\nTesting a singular matrix:")
    singular_matrix = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
    for row in singular_matrix: print(f"  {row}")
    print(f"\nResult: {calculate_inverse_4x4(singular_matrix)}")
    print("-" * 40 + "\n")

    # --- Q6: SVD Demonstration ---
    # This test decomposes a matrix A into U, S, and V^T, then reconstructs
    # it to verify that U * S * V^T is indeed equal to the original matrix A.
    print("--- Q6: SVD Demonstration ---")
    matrix_svd = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [4, 5, 6, 7]]
    print("Original Matrix (A):")
    for row in matrix_svd: print(f"  {row}")

    U_list, S_list, VT_list = jacobi_svd_4x4(matrix_svd)

    print("\nU Matrix:")
    for row in U_list: print(f"  {[f'{x:8.3f}' for x in row]}")
    print("\nS Matrix (Singular Values on Diagonal):")
    for row in S_list: print(f"  {[f'{x:8.3f}' for x in row]}")
    print("\nV Transpose Matrix:")
    for row in VT_list: print(f"  {[f'{x:8.3f}' for x in row]}")

    # Verify the reconstruction: A should equal U @ S @ V_T
    U, S, VT = np.array(U_list), np.array(S_list), np.array(VT_list)
    reconstructed = U @ S @ VT
    print("\nReconstructed Matrix (U @ S @ V.T):")
    for row in reconstructed: print(f"  {[f'{x:8.3f}' for x in row]}")

    original_as_np = np.array(matrix_svd, dtype=float)
    reconstructed_as_np = np.array(reconstructed, dtype=float)
    print(f"\nReconstruction successful: {np.allclose(original_as_np, reconstructed_as_np)}")
    print("-" * 40 + "\n")
