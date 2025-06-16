from __future__ import annotations
from itertools import product
from typing import *
import numpy as np
import lib.decompose.cnots as cnots
import lib.decompose.ising as ising

def learning_param(
    last: np.ndarray[float, 1],
    last_grad: np.ndarray[float, 1],
    cur: np.ndarray[float, 1],
    cur_grad: np.ndarray[float, 1],
) -> float:
    diff = cur - last
    diff_grad = cur_grad - last_grad
    return abs(diff @ diff_grad) / (diff_grad @ diff_grad)

ObjectiveFn = Callable[
    [np.ndarray[complex, 2], np.ndarray[float, 1]],
    float
]

GradientFn = Callable[
    [np.ndarray[complex, 2], np.ndarray[float, 1]],
    np.ndarray[float, 1]
]

# these SU(4) decompositions seem difficult to compute using conventional
# methods (e.g. actual matrix factorizations), so just do gradient ascent using
# the gate fidelity as the (inverse) objective function
def do_grad_ascent(
    U_target: np.ndarray[complex, 2],
    obj: ObjectiveFn,
    grad: GradientFn,
    init_learning_param: float,
    maxiters: Optional[int] = 1000,
    epsilon: Optional[float] = 1e-9,
) -> np.ndarray[float, 1]:
    # use a completely fixed generator here to keep decompositions consistent
    # without affecting global state, and without requiring a `seed` parameter
    gen = np.random.Generator(np.random.PCG64(10546))
    params = max(
        (2 * np.pi * gen.random(size=15) for _ in range(2000)),
        key=lambda params: obj(U_target, params)
    )
    last = params
    last_grad = grad(U_target, params)
    cur = np.copy(params)
    cur_grad = np.copy(last_grad)
    gamma = init_learning_param
    for it in range(maxiters):
        fidelity = obj(U_target, cur)
        if abs(1 - fidelity) < epsilon:
            return cur
        cur += gamma * cur_grad
        cur_grad = grad(U_target, cur)
        if (g := learning_param(last, last_grad, cur, cur_grad)):
            gamma = g
        else:
            gamma = init_learning_param
        last = np.copy(cur)
        last_grad = np.copy(cur_grad)
    print("reached maxiters")
    print(U_target)
    return params

