"""Unit tests for the oracle implementations.

Note: Qiskit uses little-endian qubit ordering.
Statevector stores |q_{n-1} ... q_0>, so target '110' (q0=1,q1=1,q2=0)
has Statevector index int('011', 2) = 3, NOT int('110', 2) = 6.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.oracles import build_oracle_v1, build_oracle_v2


def _qargs_for_target(target_state: str) -> list[int]:
    """Return qubit indices where target_state has '1' (for state preparation)."""
    return [i for i, b in enumerate(target_state) if b == '1']


def _sv_idx(bitstring: str) -> int:
    """Statevector index for a bitstring, accounting for little-endian.

    Statevector stores |q_{n-1} ... q_0>, so the index is the bitstring
    read from MSB (q_{n-1}) to LSB (q_0).
    """
    return int(bitstring[::-1], 2)


def _apply_and_measure(oracle_fn, n_qubits, target_state):
    """Build |target> then apply oracle, return statevector."""
    qc = QuantumCircuit(n_qubits)
    for i in _qargs_for_target(target_state):
        qc.x(i)
    gate = oracle_fn(n_qubits, target_state)
    qc.append(gate, range(n_qubits))
    return Statevector(qc)


class TestOracleV1:
    def test_flips_phase_on_target(self):
        """v1: Oracle should flip the phase of |101>."""
        target = '101'
        sv = _apply_and_measure(build_oracle_v1, 3, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_leaves_other_states_unchanged(self):
        """v1: States other than |101> should have zero amplitude."""
        target = '101'
        sv = _apply_and_measure(build_oracle_v1, 3, target)
        for bitstring in ['000', '001', '010', '011', '100', '110', '111']:
            idx = _sv_idx(bitstring)
            assert np.isclose(sv[idx], 0.0), f"Expected 0.0 at |{bitstring}>, got {sv[idx]}"


class TestOracleV2:
    def test_flips_phase_on_target(self):
        """v2: Oracle should flip the phase of |110>."""
        target = '110'
        sv = _apply_and_measure(build_oracle_v2, 3, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_leaves_other_states_unchanged(self):
        """v2: States other than |110> should have zero amplitude."""
        target = '110'
        sv = _apply_and_measure(build_oracle_v2, 3, target)
        for bitstring in ['000', '001', '010', '011', '100', '101', '111']:
            idx = _sv_idx(bitstring)
            assert np.isclose(sv[idx], 0.0), f"Expected 0.0 at |{bitstring}>, got {sv[idx]}"


class TestOracleEquivalence:
    def test_v1_and_v2_equivalent_on_101(self):
        """Both oracles should produce identical statevector for |101>."""
        sv1 = _apply_and_measure(build_oracle_v1, 3, '101')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '101')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_v1_and_v2_equivalent_on_110(self):
        """Both oracles should produce identical statevector for |110>."""
        sv1 = _apply_and_measure(build_oracle_v1, 3, '110')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '110')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_v1_and_v2_equivalent_on_000(self):
        """Both oracles should produce identical statevector for |000>."""
        sv1 = _apply_and_measure(build_oracle_v1, 3, '000')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '000')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_v1_and_v2_equivalent_on_111(self):
        """Both oracles should produce identical statevector for |111>."""
        sv1 = _apply_and_measure(build_oracle_v1, 3, '111')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '111')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"
