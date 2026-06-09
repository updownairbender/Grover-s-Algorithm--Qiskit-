"""Unit tests for the diffuser implementations."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
from src.diffuser import build_diffuser_v1, build_diffuser_v2


def _apply_diffuser(diffuser_fn, n_qubits, input_statevector=None):
    qc = QuantumCircuit(n_qubits)
    if input_statevector is not None:
        qc.initialize(input_statevector, range(n_qubits))
    gate = diffuser_fn(n_qubits)
    qc.append(gate, range(n_qubits))
    return Statevector(qc)


# ── Basic tests ────────────────────────────────────────────

class TestDiffuserV1:
    def test_preserves_norm(self):
        sv_in = np.ones(8) / np.sqrt(8)
        sv_out = _apply_diffuser(build_diffuser_v1, 3, sv_in)
        assert np.isclose(np.linalg.norm(sv_out), 1.0), "Norm changed after diffuser"

    def test_returns_gate(self):
        gate = build_diffuser_v1(3)
        assert isinstance(gate, Gate), "Expected Gate instance"

    def test_gate_name(self):
        gate = build_diffuser_v1(3)
        assert gate.name == 'Diffuser_v1', f"Expected 'Diffuser_v1', got '{gate.name}'"

    def test_num_qubits(self):
        gate = build_diffuser_v1(4)
        assert gate.num_qubits == 4, f"Expected 4 qubits, got {gate.num_qubits}"


class TestDiffuserV2:
    def test_preserves_norm(self):
        sv_in = np.ones(8) / np.sqrt(8)
        sv_out = _apply_diffuser(build_diffuser_v2, 3, sv_in)
        assert np.isclose(np.linalg.norm(sv_out), 1.0), "Norm changed after diffuser"

    def test_returns_gate(self):
        gate = build_diffuser_v2(3)
        assert isinstance(gate, Gate), "Expected Gate instance"

    def test_gate_name(self):
        gate = build_diffuser_v2(3)
        assert gate.name == 'Diffuser_v2', f"Expected 'Diffuser_v2', got '{gate.name}'"

    def test_num_qubits(self):
        gate = build_diffuser_v2(4)
        assert gate.num_qubits == 4, f"Expected 4 qubits, got {gate.num_qubits}"


# ── Mathematical properties ────────────────────────────────

class TestDiffuserProperties:
    def _check_magnitudes(self, sv_in, sv_out):
        """Diffuser on uniform state may add a global phase (-1)."""
        assert np.allclose(np.abs(sv_in), np.abs(sv_out.data)), \
            "Diffuser changed magnitude of uniform state"

    def test_uniform_state_unchanged_by_v1(self):
        sv_in = np.ones(8) / np.sqrt(8)
        sv_out = _apply_diffuser(build_diffuser_v1, 3, sv_in)
        self._check_magnitudes(sv_in, sv_out)

    def test_uniform_state_unchanged_by_v2(self):
        sv_in = np.ones(8) / np.sqrt(8)
        sv_out = _apply_diffuser(build_diffuser_v2, 3, sv_in)
        self._check_magnitudes(sv_in, sv_out)

    def test_uniform_4q_unchanged_by_v1(self):
        sv_in = np.ones(16) / 4.0
        sv_out = _apply_diffuser(build_diffuser_v1, 4, sv_in)
        self._check_magnitudes(sv_in, sv_out)

    def test_uniform_4q_unchanged_by_v2(self):
        sv_in = np.ones(16) / 4.0
        sv_out = _apply_diffuser(build_diffuser_v2, 4, sv_in)
        self._check_magnitudes(sv_in, sv_out)

    def test_2_qubits_v1(self):
        sv_in = np.ones(4) / 2.0
        sv_out = _apply_diffuser(build_diffuser_v1, 2, sv_in)
        self._check_magnitudes(sv_in, sv_out)

    def test_2_qubits_v2(self):
        sv_in = np.ones(4) / 2.0
        sv_out = _apply_diffuser(build_diffuser_v2, 2, sv_in)
        self._check_magnitudes(sv_in, sv_out)


# ── Equivalence ────────────────────────────────────────────

class TestDiffuserEquivalence:
    def test_equivalent_on_uniform(self):
        sv_in = np.ones(8) / np.sqrt(8)
        sv1 = _apply_diffuser(build_diffuser_v1, 3, sv_in)
        sv2 = _apply_diffuser(build_diffuser_v2, 3, sv_in)
        assert np.allclose(sv1, sv2), "v1 and v2 diffuser outputs differ"

    def test_equivalent_random_state(self):
        rng = np.random.default_rng(42)
        sv_in = rng.random(8) + 1j * rng.random(8)
        sv_in /= np.linalg.norm(sv_in)
        sv1 = _apply_diffuser(build_diffuser_v1, 3, sv_in)
        sv2 = _apply_diffuser(build_diffuser_v2, 3, sv_in)
        assert np.allclose(sv1, sv2), "v1 and v2 diffuser outputs differ"

    def test_equivalent_4q_uniform(self):
        sv_in = np.ones(16) / 4.0
        sv1 = _apply_diffuser(build_diffuser_v1, 4, sv_in)
        sv2 = _apply_diffuser(build_diffuser_v2, 4, sv_in)
        assert np.allclose(sv1, sv2), "v1 and v2 diffuser outputs differ (4q)"

    def test_equivalent_4q_random(self):
        rng = np.random.default_rng(123)
        sv_in = rng.random(16) + 1j * rng.random(16)
        sv_in /= np.linalg.norm(sv_in)
        sv1 = _apply_diffuser(build_diffuser_v1, 4, sv_in)
        sv2 = _apply_diffuser(build_diffuser_v2, 4, sv_in)
        assert np.allclose(sv1, sv2), "v1 and v2 diffuser outputs differ (4q random)"
