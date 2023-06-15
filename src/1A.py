import numpy as np
from scipy.sparse import diags
from scipy.sparse import diags, identity, kron

from scipy import sparse
from scipy.linalg import solve_triangular

def backsolv(A, b):
    """
    Written by Maximiliano Casas
    ---------------------------------------------------
    Solves an upper triangular system
    by back-substitution. 
    ---------------------------------------------------
    """
    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]

    x = x.reshape(-1)

    return x

def gauss(A, b):
    """
    Written by Maximiliano Casas
    ---------------------------------------------------
    Solves A x = b by Gauss elimination
    ---------------------------------------------------
    """
    n = A.shape[0]
    A = np.concatenate((A, b.reshape(-1, 1)), axis=1)

    for k in range(n-1):
        for i in range(k+1, n):
            piv = A[i, k] / A[k, k]
            A[i, k+1:n+1] = A[i, k+1:n+1] - piv*A[k, k+1:n+1]

    x = backsolv(A[:, :n], A[:, n])

    return x

def gaussj(A, b):
    """
    Written by Maximiliano Casas
    ---------------------------------------------------
    Solves A x = b by Gauss-Jordan elimination
    ---------------------------------------------------
    """
    n = A.shape[0]
    A = np.concatenate((A, b.reshape(-1, 1)), axis=1)

    for k in range(n):
        for i in range(n):
            if i != k:
                piv = A[i, k] / A[k, k]
                A[i, k+1:n+1] = A[i, k+1:n+1] - piv*A[k, k+1:n+1]

    x = A[:, n] / np.diag(A)

    return x


def sor(A, x0, b, maxiter, w):
    n = len(b)
    x = x0
    rs = []
    er = []
    for _ in range(maxiter):
        x_old = x.copy()
        for i in range(n):
            old_sum = np.dot(A[i, :i], x[:i])
            new_sum = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (1 - w) * x_old[i] + (w / A[i, i]) * (b[i] - old_sum - new_sum)
        rs.append(np.linalg.norm(b - np.dot(A, x)))
        er.append(np.linalg.norm(xsol - x))
    return x, rs, er


def mr(A, x, b, nsteps, tol=None):
    """
    Written by Maximiliano Casas
    Minimal residual method
    """
    r = b - np.dot(A, x)
    nrmHist = np.zeros(nsteps+1)
    nrmHist[0] = np.linalg.norm(r)
    
    if tol is None:
        tol = np.finfo(float).eps
    tol1 = tol * nrmHist[0]
    
    for i in range(nsteps):
        ar = np.dot(A, r)
        alp = np.dot(ar.T, r) / np.dot(ar.T, ar)
        x = x + alp * r
        r = r - alp * ar
        ro = np.linalg.norm(r)
        nrmHist[i+1] = ro
        if ro < tol1:
            break

    return x, nrmHist


def sor(A, x, b, nsteps, om):
    """
    Written by Maximiliano Casas
    Successive over-relaxation (SOR) method
    """
    global xsol

    D = np.diag(np.diag(A))
    E = -np.tril(A, -1)
    F = -np.triu(A, 1)
    L = D - om * E
    U = om * F + (1 - om) * D

    # record res. norm
    res0 = np.linalg.norm(b - np.dot(A, x))
    nrmHist = np.zeros(nsteps+1)
    nrmHist[0] = res0
    error = np.zeros(nsteps+1)
    error[0] = np.linalg.norm(x - xsol)

    # iterate
    for i in range(nsteps):
        x = solve_triangular(L, np.dot(U, x) + om * b, lower=True)
        nrmHist[i+1] = np.linalg.norm(b - np.dot(A, x))
        error[i+1] = np.linalg.norm(x - xsol)

    return x, nrmHist, error


def sptridiag(a, b, c, n):
    """
    Written by Maximiliano Casas
    Creates tridiagonal matrix with constants
    a, b, c respectively on the 3 diagonals
    """
    e = np.ones(n)
    T = diags([a * e, b * e, c * e], [-1, 0, 1], shape=(n, n))
    return T

    def lap2D(nx, ny):
    """
    Written by Maximiliano Casas
    Generates a 2-D Laplacian
    """
    tx = sptridiag(-1, 2, -1, nx)
    ty = sptridiag(-1, 2, -1, ny)
    A = kron(identity(ny), tx) + kron(ty, identity(nx))
    return A


def steep(A, x, b, nsteps, tol=np.finfo(float).eps):
    """
    Written by Maximiliano Casas
    Steepest descent method
    """
    r = b - np.dot(A, x)
    nrmHist = np.zeros(nsteps+1)
    nrmHist[0] = np.linalg.norm(r)
    tol1 = tol * nrmHist[0]

    # loop
    for i in range(nsteps):
        ar = np.dot(A, r)
        alp = np.dot(r.T, r) / np.dot(r.T, ar)
        x = x + alp * r
        r = r - alp * ar
        ro = np.linalg.norm(r)
        # save norms for plotting
        nrmHist[i+1] = ro
        if ro < tol1:
            break

    return x, nrmHist