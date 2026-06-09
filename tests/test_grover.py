"""Unit tests for the full Grover search algorithm.

Note: Qiskit uses little-endian qubit ordering.
counts from measure_all() return bitstrings where q0 is the rightmost
character (LSB). So target '110' (q0=1,q1=1,q2=0) appears as '011'.
"""
import math
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag
from qiskit.primitives import StatevectorSampler
from src.grover_core import grover_search
from src.config import SHOTS


def _expected_bitstring(target_state: str) -> str:
    return target_state[::-1]


# ── Functional correctness ─────────────────────────────────

class TestGroverSearch:
    def test_target_state_is_most_frequent(self):
        target = '110'
        qc = grover_search(3, target)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        assert max(counts, key=counts.get) == expected, f"Expected {expected}, got {max(counts, key=counts.get)}"

    def test_target_probability_above_90_percent(self):
        target = '101'
        qc = grover_search(3, target)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        prob = counts.get(expected, 0) / SHOTS
        assert prob > 0.9, f"Probability of |{target}> is {prob:.4f}, expected >0.9 (2 optimal iterations)"

    def test_all_3bit_targets_work(self):
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
        target = '101'
        qc = grover_search(3, target, n_iterations=1)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        prob = counts.get(expected, 0) / SHOTS
        assert 0.7 < prob < 0.9, f"Expected ~78% for 1 iteration, got {prob:.4f}"

    def test_zero_iterations(self):
        target = '101'
        qc = grover_search(3, target, n_iterations=0)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        prob = counts.get(expected, 0) / SHOTS
        assert prob < 0.2, f"With 0 iterations, |{target}> prob should be ~12.5%, got {prob:.4f}"

    def test_4_qubit_target_works(self):
        target = '1010'
        qc = grover_search(4, target)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        assert max(counts, key=counts.get) == expected, f"Grover failed for 4-qubit |{target}>"

    def test_2_qubit_target_works(self):
        target = '01'
        qc = grover_search(2, target)
        qc.measure_all()
        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=SHOTS).result()
        counts = result[0].data.meas.get_counts()
        expected = _expected_bitstring(target)
        assert max(counts, key=counts.get) == expected, f"Grover failed for 2-qubit |{target}>"

    def test_all_4bit_targets_work(self):
        targets = ['0000', '1010', '0101', '1111', '0011', '1100', '0110']
        for target in targets:
            qc = grover_search(4, target)
            qc.measure_all()
            sampler = StatevectorSampler()
            result = sampler.run([qc], shots=SHOTS).result()
            counts = result[0].data.meas.get_counts()
            expected = _expected_bitstring(target)
            assert max(counts, key=counts.get) == expected, f"Grover failed for 4q |{target}>"


# ── Circuit structure ──────────────────────────────────────

class TestGroverCircuitStructure:
    def test_returns_quantum_circuit(self):
        qc = grover_search(3, '101')
        assert isinstance(qc, QuantumCircuit), "Expected QuantumCircuit"

    def test_correct_num_qubits(self):
        qc = grover_search(4, '1010')
        assert qc.num_qubits == 4, f"Expected 4 qubits, got {qc.num_qubits}"

    def test_no_measurements_in_circuit(self):
        qc = grover_search(3, '101')
        ops = [inst.operation.name for inst in qc.data]
        assert 'measure' not in ops, "grover_search should not include measurements"

    def test_barriers_present(self):
        qc = grover_search(3, '101')
        dag = circuit_to_dag(qc)
        barrier_count = sum(1 for n in dag.op_nodes() if n.op.name == 'barrier')
        assert barrier_count > 0, "Expected at least one barrier in circuit"

    def test_correct_number_of_iterations_default(self):
        """Default iterations should be floor(π/4 * sqrt(N))."""
        qc = grover_search(3, '101')
        dag = circuit_to_dag(qc)
        gate_names = [n.op.name for n in dag.op_nodes()]
        oracle_count = sum(1 for n in gate_names if 'Oracle' in n)
        expected = int(math.pi / 4 * math.sqrt(8))
        assert oracle_count == expected, f"Expected {expected} oracle apps, got {oracle_count}"

    def test_custom_iteration_count(self):
        qc = grover_search(3, '101', n_iterations=5)
        dag = circuit_to_dag(qc)
        gate_names = [n.op.name for n in dag.op_nodes()]
        oracle_count = sum(1 for n in gate_names if 'Oracle' in n)
        assert oracle_count == 5, f"Expected 5 oracle apps, got {oracle_count}"

    def test_no_duplicate_oracle_diffuser(self):
        """Each iteration should have exactly one oracle + one diffuser gate."""
        qc = grover_search(3, '101', n_iterations=3)
        dag = circuit_to_dag(qc)
        names = [n.op.name for n in dag.op_nodes()
                 if 'Oracle' in n.op.name or 'Diffuser' in n.op.name]
        assert names == ['Oracle_v2', 'Diffuser_v2', 'Oracle_v2', 'Diffuser_v2', 'Oracle_v2', 'Diffuser_v2'], \
            f"Unexpected gate order: {names}"


# ── Optimal iteration calculation ──────────────────────────

class TestOptimalIterations:
    def test_optimal_3q(self):
        """For 3 qubits, optimal iterations = floor(π/4 * sqrt(8)) = 2."""
        expected = int(math.pi / 4 * math.sqrt(8))
        assert expected == 2, f"Expected 2 optimal iterations for 3 qubits, got {expected}"

    def test_optimal_4q(self):
        """For 4 qubits, optimal iterations = floor(π/4 * sqrt(16)) = 3."""
        expected = int(math.pi / 4 * math.sqrt(16))
        assert expected == 3, f"Expected 3 optimal iterations for 4 qubits, got {expected}"

    def test_optimal_2q(self):
        """For 2 qubits, optimal iterations = floor(π/4 * sqrt(4)) = 1."""
        expected = int(math.pi / 4 * math.sqrt(4))
        assert expected == 1, f"Expected 1 optimal iteration for 2 qubits, got {expected}"
