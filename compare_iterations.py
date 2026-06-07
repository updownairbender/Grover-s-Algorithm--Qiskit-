"""Compare Grover iteration counts: theoretical optimum vs actual accuracy."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from collections.abc import Callable
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.primitives import StatevectorSampler

from src.config import N_QUBITS, TARGET_STATE, SHOTS
from src.oracles import build_oracle_v1, build_oracle_v2
from src.diffuser import build_diffuser_v1, build_diffuser_v2


def run_grover(
    oracle_fn: Callable[[int, str], Gate],
    n_qubits: int,
    target_state: str,
    n_iter: int,
    shots: int = 4096,
) -> tuple[float, dict[str, int]]:
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    oracle = oracle_fn(n_qubits, target_state)
    diffuser = build_diffuser_v2(n_qubits)
    for _ in range(n_iter):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))
    qc.measure_all()

    sampler = StatevectorSampler()
    result = sampler.run([qc], shots=shots).result()
    counts: dict[str, int] = result[0].data.meas.get_counts()
    target_prob = counts.get(target_state, 0) / shots
    return target_prob, counts


def main():
    import os
    os.makedirs('output', exist_ok=True)

    N = 2 ** N_QUBITS
    optimal_iter = int(math.pi / 4 * math.sqrt(N))
    max_iter = optimal_iter * 2 + 1
    iterations = list(range(1, max_iter + 1))

    print(f"Qubits: {N_QUBITS} | Search space: {N} | Optimal iterations in theory: {optimal_iter}" "\n")
    print(f"{'Iter':>4} | {'v1 (MCX) prob':>14} | {'v2 (ZGate) prob':>15} | {'Delta':>7}")
    print("-" * 55)

    probs_v1, probs_v2 = [], []
    for i in iterations:
        p1, c1 = run_grover(build_oracle_v1, N_QUBITS, TARGET_STATE, i, SHOTS)
        p2, c2 = run_grover(build_oracle_v2, N_QUBITS, TARGET_STATE, i, SHOTS)
        probs_v1.append(p1)
        probs_v2.append(p2)
        print(f"{i:4d} | {p1:13.4f} | {p2:14.4f} | {p2-p1:+7.4f}")

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, probs_v1, 'o-', label='Oracle v1 (MCX)')
    plt.plot(iterations, probs_v2, 's--', label='Oracle v2 (ZGate.control)')
    plt.axvline(optimal_iter, color='gray', linestyle=':', label=f'Optimal in theory ({optimal_iter})')
    plt.xlabel('Number of iterations')
    plt.ylabel(f'Probability of |{TARGET_STATE}>')
    plt.title('Grover: Oracle comparison across iteration counts')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/iteration_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to output/iteration_comparison.png")

    # Best iteration for each
    best_v1 = iterations[probs_v1.index(max(probs_v1))]
    best_v2 = iterations[probs_v2.index(max(probs_v2))]
    print(f"Best iteration count — v1: {best_v1}, v2: {best_v2}, theory: {optimal_iter}")


if __name__ == '__main__':
    main()
