"""Diffusion operator for Grover's algorithm (inversion about the mean)."""
from collections.abc import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import ZGate


# ───────────────────────────
#  Internal helpers  (private)
# ───────────────────────────

def _apply_mcz_v1(qc: QuantumCircuit, n_qubits: int) -> None:
    """MCZ via H + MCX + H."""
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)


def _apply_mcz_v2(qc: QuantumCircuit, n_qubits: int) -> None:
    """MCZ via ZGate().control()."""
    qc.append(ZGate().control(n_qubits - 1), range(n_qubits))


def _build_diffuser(n_qubits: int, mcz_fn: Callable[[QuantumCircuit, int], None], name: str) -> Gate:
    """Shared diffuser builder: H → X → MCZ → X → H."""
    diffuser = QuantumCircuit(n_qubits, name=name)

    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))

    mcz_fn(diffuser, n_qubits)

    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    return diffuser.to_gate()


# ───────────────────────────
#  Public API  (import these)
# ───────────────────────────

def build_diffuser_v1(n_qubits: int) -> Gate:
    """Diffuser using hand-rolled MCZ (H + MCX + H)."""
    return _build_diffuser(n_qubits, _apply_mcz_v1, 'Diffuser_v1')


def build_diffuser_v2(n_qubits: int) -> Gate:
    """Diffuser using direct ZGate().control()."""
    return _build_diffuser(n_qubits, _apply_mcz_v2, 'Diffuser_v2')
