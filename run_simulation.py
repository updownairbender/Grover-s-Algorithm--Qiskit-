"""Run Grover's algorithm locally using StatevectorSampler."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.primitives import StatevectorSampler
from src.grover_core import grover_search
from src.config import N_QUBITS, TARGET_STATE, SHOTS

OUTPUT_DIR = 'output'


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    qc = grover_search(N_QUBITS, TARGET_STATE)
    qc.measure_all()

    fig_circuit = qc.draw('mpl')
    fig_circuit.savefig(f'{OUTPUT_DIR}/circuit.png', bbox_inches='tight')
    plt.close(fig_circuit)
    print(f"Circuit saved to {OUTPUT_DIR}/circuit.png")

    sampler = StatevectorSampler()
    result = sampler.run([qc], shots=SHOTS).result()
    counts: dict[str, int] = result[0].data.meas.get_counts()

    print(f"Target state: |{TARGET_STATE}>")
    print(f"Measured counts: {counts}")

    most_likely = max(counts, key=counts.get)
    status = 'correct' if most_likely == TARGET_STATE else 'wrong'
    print(f"Most frequent outcome: |{most_likely}> ({status})")

    fig_hist = plot_histogram(counts)
    fig_hist.savefig(f'{OUTPUT_DIR}/histogram.png', bbox_inches='tight')
    plt.close(fig_hist)
    print(f"Histogram saved to {OUTPUT_DIR}/histogram.png")


if __name__ == '__main__':
    main()
