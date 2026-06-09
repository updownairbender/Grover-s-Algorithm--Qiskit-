"""Unit tests for the oracle implementations.

Note: Qiskit uses little-endian qubit ordering.
Statevector stores |q_{n-1} ... q_0>, so target '110' (q0=1,q1=1,q2=0)
has Statevector index int('011', 2) = 3, NOT int('110', 2) = 6.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
from src.oracles import build_oracle_v1, build_oracle_v2


def _qargs_for_target(target_state: str) -> list[int]:
    return [i for i, b in enumerate(target_state) if b == '1']


def _sv_idx(bitstring: str) -> int:
    return int(bitstring[::-1], 2)


def _apply_and_measure(oracle_fn, n_qubits, target_state):
    qc = QuantumCircuit(n_qubits)
    for i in _qargs_for_target(target_state):
        qc.x(i)
    gate = oracle_fn(n_qubits, target_state)
    qc.append(gate, range(n_qubits))
    return Statevector(qc)


# ── Basic phase flip tests (3 qubits) ──────────────────────

class TestOracleV1:
    def test_flips_phase_on_target(self):
        target = '101'
        sv = _apply_and_measure(build_oracle_v1, 3, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_leaves_other_states_unchanged(self):
        target = '101'
        sv = _apply_and_measure(build_oracle_v1, 3, target)
        for bitstring in ['000', '001', '010', '011', '100', '110', '111']:
            idx = _sv_idx(bitstring)
            assert np.isclose(sv[idx], 0.0), f"Expected 0.0 at |{bitstring}>, got {sv[idx]}"

    def test_returns_gate(self):
        gate = build_oracle_v1(3, '101')
        assert isinstance(gate, Gate), "Expected Gate instance"

    def test_gate_name(self):
        gate = build_oracle_v1(3, '101')
        assert gate.name == 'Oracle_v1', f"Expected 'Oracle_v1', got '{gate.name}'"

    def test_target_all_zeros(self):
        target = '000'
        sv = _apply_and_measure(build_oracle_v1, 3, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_num_qubits(self):
        gate = build_oracle_v1(4, '1010')
        assert gate.num_qubits == 4, f"Expected 4 qubits, got {gate.num_qubits}"


class TestOracleV2:
    def test_flips_phase_on_target(self):
        target = '110'
        sv = _apply_and_measure(build_oracle_v2, 3, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_leaves_other_states_unchanged(self):
        target = '110'
        sv = _apply_and_measure(build_oracle_v2, 3, target)
        for bitstring in ['000', '001', '010', '011', '100', '101', '111']:
            idx = _sv_idx(bitstring)
            assert np.isclose(sv[idx], 0.0), f"Expected 0.0 at |{bitstring}>, got {sv[idx]}"

    def test_returns_gate(self):
        gate = build_oracle_v2(3, '110')
        assert isinstance(gate, Gate), "Expected Gate instance"

    def test_gate_name(self):
        gate = build_oracle_v2(3, '110')
        assert gate.name == 'Oracle_v2', f"Expected 'Oracle_v2', got '{gate.name}'"

    def test_target_all_zeros(self):
        target = '000'
        sv = _apply_and_measure(build_oracle_v2, 3, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_num_qubits(self):
        gate = build_oracle_v2(4, '1010')
        assert gate.num_qubits == 4, f"Expected 4 qubits, got {gate.num_qubits}"


# ── 4-qubit tests ──────────────────────────────────────────

class TestOracle4Qubits:
    def test_v1_flips_4q(self):
        target = '1010'
        sv = _apply_and_measure(build_oracle_v1, 4, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"v1: Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_v2_flips_4q(self):
        target = '0101'
        sv = _apply_and_measure(build_oracle_v2, 4, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"v2: Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_v1_v2_equivalent_4q(self):
        target = '1100'
        sv1 = _apply_and_measure(build_oracle_v1, 4, target)
        sv2 = _apply_and_measure(build_oracle_v2, 4, target)
        assert np.allclose(sv1, sv2), "v1 and v2 differ on 4 qubits"

    def test_v1_leaves_others_4q(self):
        target = '1010'
        sv = _apply_and_measure(build_oracle_v1, 4, target)
        for bitstring in ['0000', '1111', '0101']:
            idx = _sv_idx(bitstring)
            assert np.isclose(sv[idx], 0.0), f"Expected 0.0 at |{bitstring}>, got {sv[idx]}"


# ── Equivalence across targets ─────────────────────────────

class TestOracleEquivalence:
    def test_equivalent_101(self):
        sv1 = _apply_and_measure(build_oracle_v1, 3, '101')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '101')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_equivalent_110(self):
        sv1 = _apply_and_measure(build_oracle_v1, 3, '110')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '110')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_equivalent_000(self):
        sv1 = _apply_and_measure(build_oracle_v1, 3, '000')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '000')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_equivalent_111(self):
        sv1 = _apply_and_measure(build_oracle_v1, 3, '111')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '111')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_equivalent_010(self):
        sv1 = _apply_and_measure(build_oracle_v1, 3, '010')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '010')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"

    def test_equivalent_001(self):
        sv1 = _apply_and_measure(build_oracle_v1, 3, '001')
        sv2 = _apply_and_measure(build_oracle_v2, 3, '001')
        assert np.allclose(sv1, sv2), "v1 and v2 statevectors differ"


# ── Edge cases ─────────────────────────────────────────────

class TestOracleEdgeCases:
    def test_2_qubits_v1(self):
        target = '10'
        sv = _apply_and_measure(build_oracle_v1, 2, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"v1 2q: Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_2_qubits_v2(self):
        target = '01'
        sv = _apply_and_measure(build_oracle_v2, 2, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"v2 2q: Expected -1.0 at |{target}>, got {sv[idx]}"

    def test_v2_all_ones(self):
        target = '111'
        sv = _apply_and_measure(build_oracle_v2, 3, target)
        idx = _sv_idx(target)
        assert np.isclose(sv[idx], -1.0), f"Expected -1.0 at |{target}>, got {sv[idx]}"
