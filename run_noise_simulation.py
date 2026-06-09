"""
Run Grover's algorithm with a realistic noise model (AerSimulator + GenericBackendV2).

Difference from run_simulation.py:
- run_simulation.py uses StatevectorSampler (ideal simulation, no noise).
  Result: probability of target state is *near* 100%.
- run_noise_simulation.py uses AerSimulator with GenericBackendV2 (IBM-like noise).
  Result: probability distribution affected by gate errors, T1/T2, readout errors
  — closer to actual IBM hardware output.
  This demonstrates understanding of the gap between ideal simulation and real hardware
  without using actual IBM hardware.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from src.config import N_QUBITS, TARGET_STATE, SHOTS
from src.grover_core import grover_search

OUTPUT_DIR = 'output/noise_simulation'


def _build_noisy_backend():
    """Create a FakeBackend that mimics real IBM noise."""
    backend = GenericBackendV2(
        num_qubits=N_QUBITS,
        coupling_map=None,
        basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'],
        seed=42,
    )
    return backend


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # build the circuit
    qc = grover_search(N_QUBITS, TARGET_STATE)
    qc.measure_all()
    qc.name = f'grover_noise_{N_QUBITS}q_{TARGET_STATE}'

    # save circuit figure before transpilation
    fig = qc.draw('mpl')
    fig.savefig(f'{OUTPUT_DIR}/circuit.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Circuit saved to {OUTPUT_DIR}/circuit.png")

    # create noisy backend
    backend = _build_noisy_backend()
    print(f"Backend: {backend.name}")
    print(f"  num_qubits: {backend.num_qubits}")
    print(f"  basis_gates: {backend._basis_gates}")

    # transpile circuit for backend compatibility
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1,
        seed_transpiler=42,
    )
    qc_transpiled = pm.run(qc)

    # save transpiled circuit figure
    fig2 = qc_transpiled.draw('mpl')
    fig2.savefig(f'{OUTPUT_DIR}/circuit_transpiled.png', bbox_inches='tight')
    plt.close(fig2)
    print(f"Transpiled circuit saved to {OUTPUT_DIR}/circuit_transpiled.png")

    # run noise simulation
    noise_sim = AerSimulator.from_backend(backend)
    job = noise_sim.run([qc_transpiled], shots=SHOTS)
    result = job.result()
    counts = result.get_counts(0)

    # save results to text file
    with open(f'{OUTPUT_DIR}/results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Target state: |{TARGET_STATE}>\n")
        f.write(f"Shots: {SHOTS}\n")
        f.write(f"Backend: {backend.name}\n\n")
        f.write("Measured counts:\n")
        for state, count in sorted(counts.items(), key=lambda x: -x[1]):
            f.write(f"  |{state}>: {count} ({100*count/SHOTS:.1f}%)\n")

    # print results to console
    print(f"\nTarget state: |{TARGET_STATE}>")
    print(f"Measured counts:")
    total = 0
    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / SHOTS
        print(f"  |{state}>: {count} ({pct:.1f}%)")
        total += count
    print(f"  Total: {total}")

    # verify target state success rate
    target_count = counts.get(TARGET_STATE, 0)
    target_pct = 100 * target_count / SHOTS
    print(f"\n|{TARGET_STATE}> success rate: {target_pct:.1f}%")

    # plot histogram
    fig_hist = plot_histogram(counts, number_to_keep=8)
    fig_hist.savefig(f'{OUTPUT_DIR}/histogram.png', bbox_inches='tight')
    plt.close(fig_hist)
    print(f"Histogram saved to {OUTPUT_DIR}/histogram.png")
    print(f"Results saved to {OUTPUT_DIR}/results.txt")


if __name__ == '__main__':
    main()
