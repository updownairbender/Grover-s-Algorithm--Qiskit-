"""Run Grover's algorithm on a real IBM Quantum backend."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from src.grover_core import grover_search
from src.config import N_QUBITS, TARGET_STATE, SHOTS, IBM_CHANNEL, IBM_TOKEN, IBM_INSTANCE, IBM_BACKEND

OUTPUT_DIR = 'output'

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
except ImportError:
    print("qiskit-ibm-runtime is required. Install it with:")
    print("  pip install qiskit-ibm-runtime")
    raise


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not IBM_TOKEN:
        print("ERROR: IBM_TOKEN is not set.")
        print("Set it as an environment variable or in a .env file.")
        print("See: https://quantum.ibm.com/ -> Account -> API token")
        return

    print(f"Connecting to IBM Quantum ({IBM_CHANNEL})...")
    service = QiskitRuntimeService(
        channel=IBM_CHANNEL,
        token=IBM_TOKEN,
        instance=IBM_INSTANCE,
    )
    backend = service.backend(IBM_BACKEND)
    print(f"Backend: {backend.name}")

    qc = grover_search(N_QUBITS, TARGET_STATE)
    qc.measure_all()

    fig_circuit = qc.draw('mpl')
    fig_circuit.savefig(f'{OUTPUT_DIR}/circuit.png', bbox_inches='tight')
    plt.close(fig_circuit)
    print(f"Circuit saved to {OUTPUT_DIR}/circuit.png")

    print("Submitting job to IBM Quantum...")
    sampler = Sampler(mode=backend)
    job = sampler.run([qc], shots=SHOTS)
    print(f"Job ID: {job.job_id()}")

    result = job.result()
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
