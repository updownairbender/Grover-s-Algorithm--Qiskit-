import math
from qiskit import QuantumCircuit


def build_oracle(n_qubits, target_state):
    """Phase oracle that tags |target_state> with a negative phase.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    target_state : str
        Bitstring marking the target state, e.g. '101'.

    Returns
    -------
    Gate
        The oracle gate.
    """
    oracle = QuantumCircuit(n_qubits, name='Oracle')

    for i, bit in enumerate(target_state):
        if bit == '0':
            oracle.x(i)

    oracle.h(n_qubits - 1)
    oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    oracle.h(n_qubits - 1)

    for i, bit in enumerate(target_state):
        if bit == '0':
            oracle.x(i)

    return oracle.to_gate()


def build_diffuser(n_qubits):
    """Grover diffusion operator (inversion about the mean).

    H^n (2|0><0| - I) H^n implemented via X + MCZ + X.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    Gate
        The diffuser gate.
    """
    diffuser = QuantumCircuit(n_qubits, name='Diffuser')

    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))

    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)

    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    return diffuser.to_gate()


def grover_search(n_qubits, target_state):
    """Build the complete Grover search circuit (without measurements).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    target_state : str
        Bitstring to search for, e.g. '101'.

    Returns
    -------
    QuantumCircuit
        The full Grover circuit.
    """
    qc = QuantumCircuit(n_qubits)

    qc.h(range(n_qubits))

    N = 2 ** n_qubits
    n_iterations = int(math.pi / 4 * math.sqrt(N))

    oracle = build_oracle(n_qubits, target_state)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))

    return qc
