"""Oracle implementations for Grover's algorithm."""
from collections.abc import Callable

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit.circuit import Gate


# ───────────────────────────
#  Internal helpers  (private)
# ───────────────────────────

def _flip_zeros(qc: QuantumCircuit, target_state: str) -> None:
    """Apply X on qubits that are 0 in target_state."""
    for i, bit in enumerate(target_state):
        if bit == '0':
            qc.x(i)


def _apply_mcz_v1(qc: QuantumCircuit, n_qubits: int) -> None:
    """Multi-controlled Z via H + MCX + H"""
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)


def _apply_mcz_v2(qc: QuantumCircuit, n_qubits: int) -> None:
    """Multi-controlled Z via ZGate().control()"""
    qc.append(ZGate().control(n_qubits - 1), range(n_qubits))


def _build_oracle(
    n_qubits: int,
    target_state: str,
    mcz_fn: Callable[[QuantumCircuit, int], None],
    name: str,
) -> Gate:
    """Shared oracle builder: flip zeros → MCZ → flip zeros"""
    oracle = QuantumCircuit(n_qubits, name=name)
    _flip_zeros(oracle, target_state)
    mcz_fn(oracle, n_qubits)
    _flip_zeros(oracle, target_state)
    return oracle.to_gate()


# ───────────────────────────
#  Public API  (import these)
# ───────────────────────────

def build_oracle_v1(n_qubits: int, target_state: str) -> Gate:
    """Oracle using hand-rolled MCZ (H + MCX + H)."""
    return _build_oracle(n_qubits, target_state, _apply_mcz_v1, 'Oracle_v1')


def build_oracle_v2(n_qubits: int, target_state: str) -> Gate:
    """Oracle using direct ZGate().control()."""
    return _build_oracle(n_qubits, target_state, _apply_mcz_v2, 'Oracle_v2')
