from __future__ import annotations
from abc import (abstractmethod, abstractstaticmethod)
from dataclasses import dataclass
import random
from typing import *
import numpy as np
from lib.decompose import do_grad_ascent
import lib.decompose.cnots as cnots
import lib.decompose.ising as ising

class Gate:
    """
    Base class for a single unitary gate operation.
    """
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def controls(self) -> list[int]:
        pass

    @abstractmethod
    def targets(self) -> list[int]:
        pass

    @abstractmethod
    def args(self) -> list[float]:
        pass

@dataclass
class Hadamard(Gate):
    """
    A single Hadamard gate.
    """
    target: int

    def name(self) -> str:
        return "h"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class S(Gate):
    """
    A single S = sqrt(Z) gate.
    """
    target: int

    def name(self) -> str:
        return "s"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class Sdag(Gate):
    """
    A single S^-1 = conj(S) gate.
    """
    target: int

    def name(self) -> str:
        return "sdag"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class Cnot(Gate):
    """
    A single CNOT gate.
    """
    control: int
    target: int

    def name(self) -> str:
        return "cnot"

    def controls(self) -> list[int]:
        return [self.control]

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class Rx(Gate):
    """
    A single rotation about X.
    """
    target: int
    angle: float

    def name(self) -> str:
        return "rx"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle]

@dataclass
class Ry(Gate):
    """
    A single rotation about Y.
    """
    target: int
    angle: float

    def name(self) -> str:
        return "ry"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle]

@dataclass
class Rz(Gate):
    """
    A single rotation about Z.
    """
    target: int
    angle: float

    def name(self) -> str:
        return "rz"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle]

@dataclass
class U3(Gate):
    """
    A single single-qubit rotation gate with 3 Euler angles:
        U3(theta, phi, lambda) =
            Rz(phi - pi/2) Rx(pi/2) Rz(pi - theta) Rx(pi/2) Rz(lambda - pi/2)
    """
    target: int
    angle0: float
    angle1: float
    angle2: float

    def name(self) -> str:
        return "u3"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle0, self.angle1, self.angle2]

    def to_pauli_rots(self) -> list[Gate]:
        return [
            Rz(self.target, self.angle2 - np.pi / 2),
            Rx(self.target, np.pi / 2),
            Rz(self.target, np.pi - self.angle0),
            Rx(self.target, np.pi / 2),
            Rz(self.target, self.angle1 - np.pi / 2),
        ]

@dataclass
class Rxx(Gate):
    """
    A single two-qubit rotation about XX.
    """
    target0: int
    target1: int
    angle: float

    def name(self) -> str:
        return "rxx"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target0, self.target1]

    def args(self) -> list[float]:
        return [self.angle]

    def to_cnots(self) -> list[Gate]:
        return [
            Hadamard(self.target0),
            Hadamard(self.target1),
            Cnot(self.target0, self.target1),
            Rz(self.target1, self.angle),
            Cnot(self.target0, self.target1),
            Hadamard(self.target0),
            Hadamard(self.target1),
        ]

@dataclass
class Ryy(Gate):
    """
    A single two-qubit rotation about YY.
    """
    target0: int
    target1: int
    angle: float

    def name(self) -> str:
        return "ryy"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target0, self.target1]

    def args(self) -> list[float]:
        return [self.angle]

    def to_cnots(self) -> list[Gate]:
        return [
            Sdag(self.target0),
            Hadamard(self.target0),
            S(self.target0),
            Sdag(self.target1),
            Hadamard(self.target1),
            S(self.target1),
            Cnot(self.target0, self.target1),
            Rz(self.target1, self.angle),
            Cnot(self.target0, self.target1),
            Sdag(self.target0),
            Hadamard(self.target0),
            S(self.target0),
            Sdag(self.target1),
            Hadamard(self.target1),
            S(self.target1),
        ]

@dataclass
class Rzz(Gate):
    """
    A single two-qubit rotation about ZZ.
    """
    target0: int
    target1: int
    angle: float

    def name(self) -> str:
        return "rzz"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target0, self.target1]

    def args(self) -> list[float]:
        return [self.angle]

    def to_cnots(self) -> list[Gate]:
        return [
            Cnot(self.target0, self.target1),
            Rz(self.target1, self.angle),
            Cnot(self.target0, self.target1),
        ]

class SU4Decomp:
    """
    Base class for decompositions of a general SU(4) (i.e. two-qubit) unitary
    gate.
    """
    @abstractstaticmethod
    def from_uni(uni: np.ndarray[complex, 2]):
        pass

    @abstractmethod
    def to_gates(self, target0: int, target1: int) -> list[Gate]:
        pass

class IsingDecomp(SU4Decomp):
    """
    An "Ising-like" decomposition of a general SU(4) gate according to [this
    form][pennylane]:
        U =
            U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
            R_XX([eta0]) R_YY([eta1]) R_ZZ([eta2])
            U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])

    [pennylane]: https://pennylane.ai/qml/demos/tutorial_kak_decomposition#kokcu-fdhs
    """
    # [
    #   alpha0, .., alpha2,
    #   beta0, .., beta2,
    #   eta0, .., eta2,
    #   gamma0, .., gamma2,
    #   delta0, .., delta2
    # ]
    params: np.ndarray[float]

    def __init__(self, params: np.ndarray[float, 1]):
        if params.shape != (15,):
            raise ValueError("requires 15 params")
        self.params = params

    @staticmethod
    def from_uni(uni: np.ndarray[complex, 2]):
        params = do_grad_ascent(
            U_target=uni,
            obj=ising.fidelity,
            grad=ising.fidelity_grad,
            init_learning_param=1e-5,
            maxiters=5000,
            epsilon=1e-6,
        )
        return IsingDecomp(params)

    def to_gates(self, target0: int, target1: int) -> list[Gate]:
        return [
            U3(target0, *self.params[9:12]),
            U3(target1, *self.params[12:15]),
            Rxx(target0, target1, self.params[6]),
            Ryy(target0, target1, self.params[7]),
            Rzz(target0, target1, self.params[8]),
            U3(target0, *self.params[0:3]),
            U3(target1, *self.params[3:6]),
        ]

class CnotsDecomp(SU4Decomp):
    """
    A "CNOT-based" decomposition of a general SU(4) gate according to [this
    form][cnot-based]:
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
    # [
    #   alpha0, .., alpha2,
    #   beta0, .., beta2,
    #   eta0, .., eta2,
    #   gamma0, .., gamma2,
    #   delta0, .., delta2
    # ]
    params: np.ndarray[float]

    def __init__(self, params: np.ndarray[float, 1]):
        if params.shape != (15,):
            raise ValueError("requires 15 params")
        self.params = params

    @staticmethod
    def from_uni(uni: np.ndarray[complex, 2]):
        params = do_grad_ascent(
            U_target=uni,
            obj=cnots.fidelity,
            grad=cnots.fidelity_grad,
            init_learning_param=1e-5,
            maxiters=5000,
            epsilon=1e-6,
        )
        return CnotsDecomp(params)

    def to_gates(self, target0: int, target1: int) -> list[Gate]:
        return [
            Rz(target1, self.params[8]),
            Cnot(target0, target1),
            U3(target0, *self.params[9:12]),
            U3(target1, *self.params[12:15]),
            Cnot(target0, target1),
            Rx(target0, self.params[6]),
            Rz(target1, self.params[7]),
            Cnot(target0, target1),
            U3(target0, *self.params[0:3]),
            U3(target1, *self.params[3:6]),
        ]

@dataclass
class SU4:
    """
    A two-qubit unitary matrix.
    """
    target0: int
    target1: int
    mat: np.ndarray[complex, 2]

@dataclass
class PeakedCircuit:
    """
    A circuit containing only unitary operations, producing a computational
    basis state with modal probability for an assumed all-zero initial state.
    """
    seed: int
    gen: np.random.Generator
    num_qubits: int
    gates: list[Gate]
    target_state: str # all 0's and 1's
    peak_prob: float

    @staticmethod
    def from_su4_series(
        target_state: str,
        peak_prob,
        unis: list[SU4],
        seed: int,
    ):
        num_qubits = 0
        gates = list()
        gen = np.random.Generator(np.random.PCG64(seed))
        print("convert to ordinary gates:")
        n = len(unis)
        print(f"  0 / {n} ", flush=True, end="")
        for (k, uni) in enumerate(unis):
            print(f"\r  {k} / {n} ", flush=True, end="")
            num_qubits = max(num_qubits, uni.target0, uni.target1)
            if gen.random() < 0.5:
                gates += (
                    IsingDecomp.from_uni(uni.mat)
                    .to_gates(uni.target0, uni.target1)
                )
            else:
                gates += (
                    CnotsDecomp.from_uni(uni.mat)
                    .to_gates(uni.target0, uni.target1)
                )
        assert num_qubits <= len(target_state)
        return PeakedCircuit(
            seed,
            gen,
            num_qubits,
            gates,
            target_state,
            peak_prob,
        )

    # really dumb rendering because the circuits are pretty simple
    def to_qasm(self) -> str:
        """
        Render to a bare string giving the OpenQASM (2.0) circuit.
        """
        acc = (
f"""
OPENQASM 2.0;
include "qelib1.inc";

qreg q[{self.num_qubits}];

"""
        )
        for gate in self.gates:
            if isinstance(gate, U3) and self.gen.random() < 0.5:
                decomp = gate.to_pauli_rots()
                for subgate in decomp:
                    acc += write_gate(subgate)
            elif isinstance(gate, (Rxx, Ryy, Rzz)) and self.gen.random() < 0.5:
                decomp = gate.to_cnots()
                for subgate in decomp:
                    acc += write_gate(subgate)
            else:
                acc += write_gate(gate)
        return acc

def write_gate(gate: Gate) -> str:
    acc = gate.name()
    if (n := len(args := gate.args())) > 0:
        acc += "("
        acc += ",".join(f"{arg}" for arg in args)
        acc += ")"
    operands = gate.controls() + gate.targets()
    n = len(operands)
    assert n > 0, "unexpected gate with no operands"
    acc += " "
    acc += ",".join(f"q[{op}]" for op in operands)
    acc += ";\n"
    return acc

