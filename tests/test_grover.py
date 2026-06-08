"""Unit tests for the full Grover search algorithm.

Note: Qiskit uses little-endian qubit ordering.
counts from measure_all() return bitstrings where q0 is the rightmost
character (LSB). So target '110' (q0=1,q1=1,q2=0) appears as '011'.
"""
import math
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from src.grover_core import grover_search
from src.config import SHOTS


def _expected_bitstring(target_state: str) -> str:
    """Convert target_state to Qiskit's measurement bitstring convention.

    grover_search uses target_state[i] as qubit i. Qiskit's measure_all()
    returns bitstrings with q0 as the rightmost (LSB) character.
    """
    return target_state[::-1]


class TestGroverSearch:
    def test_target_state_is_most_frequent(self):
        """Grover should amplify |110> as the most frequent outcome."""
        target = '110'
        qc = grover_search(3, target)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        assert max(counts, key=counts.get) == expected, f"Expected {expected}, got {max(counts, key=counts.get)}"

    def test_target_probability_above_90_percent(self):
        """Grover should amplify |101> to >90% probability."""
        target = '101'
        qc = grover_search(3, target)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        prob = counts.get(expected, 0) / SHOTS
        assert prob > 0.9, f"Probability of |{target}> is {prob:.4f}, expected >0.9 (2 optimal iterations)"

    def test_different_targets_work(self):
        """Grover should find any 3-bit target."""
        targets = ['000', '011', '111', '110', '101', '100', '010', '001']
        for target in targets:
            qc = grover_search(3, target)
            qc.measure_all()
            sampler = StatevectorSampler()
            result = sampler.run([qc], shots=SHOTS).result()
            counts = result[0].data.meas.get_counts()
            expected = _expected_bitstring(target)
            assert max(counts, key=counts.get) == expected, f"Grover failed for |{target}> (expected {expected}, got {max(counts, key=counts.get)})"

    def test_custom_iterations(self):
        """n_iterations parameter should be respected."""
        target = '101'
        qc = grover_search(3, target, n_iterations=1)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        prob = counts.get(expected, 0) / SHOTS
        assert 0.7 < prob < 0.9, f"Expected ~78% for 1 iteration, got {prob:.4f}"

    def test_invalid_target_doesnt_crash(self):
        """build_oracle should not crash for valid inputs."""
        from src.oracles import build_oracle_v2 as build_oracle
        try:
            gate = build_oracle(3, '101')
            assert gate is not None
        except Exception as e:
            assert False, f"build_oracle raised an exception: {e}"
