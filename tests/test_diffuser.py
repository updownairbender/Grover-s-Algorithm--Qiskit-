"""Unit tests for the diffuser implementations."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.diffuser import build_diffuser_v1, build_diffuser_v2


def _apply_diffuser(diffuser_fn, n_qubits, input_statevector=None):
    """Apply diffuser and return the resulting statevector."""
    qc = QuantumCircuit(n_qubits)
    if input_statevector is not None:
        qc.initialize(input_statevector, range(n_qubits))
    gate = diffuser_fn(n_qubits)
    qc.append(gate, range(n_qubits))
    return Statevector(qc)


class TestDiffuserV1:
    def test_diffuser_v1_preserves_norm(self):
        """v1: Applying diffuser should preserve statevector norm."""
        sv_in = np.ones(8) / np.sqrt(8)  # uniform superposition
        sv_out = _apply_diffuser(build_diffuser_v1, 3, sv_in)
        assert np.isclose(np.linalg.norm(sv_out), 1.0), "Norm changed after diffuser"


class TestDiffuserV2:
    def test_diffuser_v2_preserves_norm(self):
        """v2: Applying diffuser should preserve statevector norm."""
        sv_in = np.ones(8) / np.sqrt(8)
        sv_out = _apply_diffuser(build_diffuser_v2, 3, sv_in)
        assert np.isclose(np.linalg.norm(sv_out), 1.0), "Norm changed after diffuser"


class TestDiffuserEquivalence:
    def test_v1_and_v2_equivalent_on_uniform(self):
        """Both diffusers should produce identical output from uniform superposition."""
        sv_in = np.ones(8) / np.sqrt(8)
        sv1 = _apply_diffuser(build_diffuser_v1, 3, sv_in)
        sv2 = _apply_diffuser(build_diffuser_v2, 3, sv_in)
        assert np.allclose(sv1, sv2), "v1 and v2 diffuser outputs differ"

    def test_v1_and_v2_equivalent_random_state(self):
        """Both diffusers should produce identical output from a random state."""
        rng = np.random.default_rng(42)
        sv_in = rng.random(8) + 1j * rng.random(8)
        sv_in /= np.linalg.norm(sv_in)
        sv1 = _apply_diffuser(build_diffuser_v1, 3, sv_in)
        sv2 = _apply_diffuser(build_diffuser_v2, 3, sv_in)
        assert np.allclose(sv1, sv2), "v1 and v2 diffuser outputs differ"
