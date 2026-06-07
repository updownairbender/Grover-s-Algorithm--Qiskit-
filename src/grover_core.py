import math
from qiskit import QuantumCircuit
from src.oracles import build_oracle_v2 as build_oracle
from src.diffuser import build_diffuser_v2 as build_diffuser


def grover_search(
    n_qubits: int,
    target_state: str,
    n_iterations: int | None = None,
) -> QuantumCircuit:
    """Build the complete Grover search circuit (without measurements).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    target_state : str
        Bitstring to search for, e.g. '101'.
    n_iterations : int or None
        Number of Grover iterations. If None, uses the optimal
        theoretical value: floor(π/4 * sqrt(2^n)).

    Returns
    -------
    QuantumCircuit
        The full Grover circuit.
    """
    qc = QuantumCircuit(n_qubits)

    qc.h(range(n_qubits))

    if n_iterations is None:
        N = 2 ** n_qubits
        n_iterations = int(math.pi / 4 * math.sqrt(N))

    oracle = build_oracle(n_qubits, target_state)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations - 1):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))

    return qc
