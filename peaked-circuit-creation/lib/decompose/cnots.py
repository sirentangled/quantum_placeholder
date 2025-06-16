import numpy as np
from numpy import (sin, cos, exp, kron, conjugate)

"""
Functions in this module compute quantities relevant to a "CNOT-based"
decomposition of a general SU(4) gate according to [this form][cnot-based]:
    U =
        U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
        CNOT
        Rx(0, eta0) Rz(1, eta1)
        CNOT
        U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])
        CNOT
        Rz(1, eta2)

[cnot-based]: https://arxiv.org/abs/quant-ph/0308033
"""

def u3(alpha: float, beta: float, gamma: float) -> np.ndarray[complex, 2]:
    return np.array([
        [
            cos(alpha),
            -exp(1j * gamma) * sin(alpha),
        ],
        [
            exp(1j * beta) * sin(alpha),
            exp(1j * (beta + gamma)) * cos(alpha),
        ],
    ])

def rx(angle: float) -> np.ndarray[complex, 2]:
    return np.array([
        [ cos(angle / 2), -1j * sin(angle / 2), ],
        [ -1j * sin(angle / 2), cos(angle / 2), ],
    ])

def rz(angle: float) -> np.ndarray[complex, 2]:
    return np.array([
        [ exp(-1j * angle / 2), 0 ],
        [ 0, exp( 1j * angle / 2) ],
    ])

def make_uni(params: np.ndarray[float, 1]) -> np.ndarray[complex, 2]:
    """
    Compute the full unitary matrix following the decomposition form above.
    """
    [
        alpha0, alpha1, alpha2,
        beta0, beta1, beta2,
        eta0, eta1, eta2,
        gamma0, gamma1, gamma2,
        delta0, delta1, delta2
    ] = params
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    return (
        kron(u3(alpha0, alpha1, alpha2), u3(beta0, beta1, beta2))
        @ cnot
        @ kron(rx(eta0), rz(eta1))
        @ cnot
        @ kron(u3(gamma0, gamma1, gamma2), u3(delta0, delta1, delta2))
        @ cnot
        @ kron(np.eye(2), rz(eta2))
    )

def fidelity(
    U_target: np.ndarray[complex, 2],
    params: np.ndarray[float, 1],
) -> float:
    r"""
    Compute the gate fidelity
        F = |Tr(U^\dagger U')|^2 / 16
    where U' is the unitary to decompose and U (`U_target`) is the result of
    plugging `params` into the "Ising-like" decomposition above. `params` is
    expected as
        params = [
            alpha0, .., alpha2,
            beta0, .., beta2,
            eta0, .., eta2,
            gamma0, .., gamma2,
            delta0, .., delta2,
        ]
    """
    uni = make_uni(params)
    return abs(np.diag(uni.T.conjugate() @ U_target).sum()) ** 2 / 16

def step(
    U_target: np.ndarray[complex, 2],
    params: np.ndarray[float, 1],
    stepsize: float,
    pos: int,
) -> float:
    params[pos] += stepsize
    f_plus = fidelity(U_target, params)
    params[pos] -= 2 * stepsize
    f_minus = fidelity(U_target, params)
    params[pos] += stepsize
    return (f_plus - f_minus) / (2 * stepsize)

def fidelity_grad(
    U_target: np.ndarray[complex, 2],
    params: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    """
    Compute the gradient of the fidelity with respect to all parameters.
    """
    return np.array([step(U_target, params, 1e-6, k) for k in range(15)])

