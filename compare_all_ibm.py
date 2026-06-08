"""Compare all oracle/diffuser combinations on a real IBM Quantum backend.

Tests four combinations: v1/v2 oracles × v1/v2 diffusers.
Each execution creates a numbered run folder (run_001/, run_002/, …)
containing:
  - results.json           raw counts for all four combos
  - comparison_bars.png    bar chart comparing target probabilities
  - O2+D2_circuit.png      circuit diagram (last combo only)
  - *_histogram.png        measurement histogram for each combo

At the base level, a cross-run summary is updated every execution:
  - summary.json           aggregated probabilities from all runs
  - summary_chart.png      line chart tracking each combo across runs
"""

import os
import re
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.visualization import plot_histogram

from src.config import (
    N_QUBITS, TARGET_STATE, SHOTS,
    IBM_CHANNEL, IBM_TOKEN, IBM_INSTANCE, IBM_BACKEND,
)
from src.oracles import build_oracle_v1, build_oracle_v2
from src.diffuser import build_diffuser_v1, build_diffuser_v2

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
except ImportError:
    print("qiskit-ibm-runtime is required. Install it with:")
    print("  pip install qiskit-ibm-runtime")
    raise


BASE_DIR = 'output/ibm_comparison'

OracleFn = Callable[[int, str], Gate]
DiffuserFn = Callable[[int], Gate]

COMBOS: list[tuple[str, OracleFn, DiffuserFn]] = [
    ('O1+D1', build_oracle_v1, build_diffuser_v1),
    ('O1+D2', build_oracle_v1, build_diffuser_v2),
    ('O2+D1', build_oracle_v2, build_diffuser_v1),
    ('O2+D2', build_oracle_v2, build_diffuser_v2),
]


@dataclass
class Result:
    """Holds the outcome of one combination run."""
    label: str
    counts: dict[str, int]
    target_prob: float
    job_id: str
    qc: QuantumCircuit


def _next_run_dir() -> str:
    """Create and return the next numbered run directory (run_001, run_002, …)."""
    os.makedirs(BASE_DIR, exist_ok=True)
    existing = [d for d in os.listdir(BASE_DIR) if re.fullmatch(r'run_\d{3}', d)]
    n = 1 + (max(int(d.split('_')[1]) for d in existing) if existing else 0)
    path = os.path.join(BASE_DIR, f'run_{n:03d}')
    os.makedirs(path)
    return path


def build_circuit(oracle_fn: OracleFn, diffuser_fn: DiffuserFn) -> QuantumCircuit:
    """Build a Grover circuit for the given oracle/diffuser pair."""
    qc = QuantumCircuit(N_QUBITS)
    qc.h(range(N_QUBITS))
    oracle = oracle_fn(N_QUBITS, TARGET_STATE)
    diffuser = diffuser_fn(N_QUBITS)
    for _ in range(2):
        qc.append(oracle, range(N_QUBITS))
        qc.append(diffuser, range(N_QUBITS))
    qc.measure_all()
    return qc


def save_circuit(qc: QuantumCircuit, label: str, path: str) -> None:
    """Save circuit diagram as PNG."""
    fig = qc.draw('mpl')
    fig.savefig(os.path.join(path, f'{label}_circuit.png'), bbox_inches='tight')
    plt.close(fig)


def save_histogram(counts: dict[str, int], label: str, path: str) -> None:
    """Save measurement histogram as PNG."""
    fig = plot_histogram(counts)
    fig.savefig(os.path.join(path, f'{label}_histogram.png'), bbox_inches='tight')
    plt.close(fig)


def save_comparison_chart(results: list[Result], path: str) -> None:
    """Save a grouped bar chart comparing target probabilities."""
    labels = [r.label for r in results]
    probs = [r.target_prob for r in results]
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, probs, color=colors, edgecolor='black')

    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{prob:.3f}', ha='center', fontsize=11)

    plt.axhline(1 / 2**N_QUBITS, color='gray', linestyle='--', label=f'Random guess (1/{2**N_QUBITS})')
    plt.ylabel(f'Probability of |{TARGET_STATE}>')
    plt.title(f'Grover: Oracle/Diffuser Comparison on {IBM_BACKEND}')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'comparison_bars.png'), dpi=150)
    plt.close()


def main() -> None:
    if not IBM_TOKEN:
        print("ERROR: IBM_TOKEN is not set.")
        return

    run_dir = _next_run_dir()
    print(f"Run directory: {run_dir}\n")

    print(f"Connecting to IBM Quantum ({IBM_CHANNEL})...")
    service = QiskitRuntimeService(
        channel=IBM_CHANNEL,
        token=IBM_TOKEN,
        instance=IBM_INSTANCE,
    )
    backend = service.backend(IBM_BACKEND)
    print(f"Backend: {backend.name}\n")

    sampler = Sampler(mode=backend)

    results: list[Result] = []

    for label, oracle_fn, diffuser_fn in COMBOS:
        qc = build_circuit(oracle_fn, diffuser_fn)

        # save circuit diagram only for the last combination
        if label == COMBOS[-1][0]:
            save_circuit(qc, label, run_dir)
            print(f"Saved {label}_circuit.png")

        print(f"Submitting {label} to IBM Quantum...")
        job = sampler.run([qc], shots=SHOTS)
        result = job.result()

        counts: dict[str, int] = result[0].data.meas.get_counts()
        target_prob = counts.get(TARGET_STATE, 0) / SHOTS
        results.append(Result(label, counts, target_prob, job.job_id(), qc))

        save_histogram(counts, label, run_dir)
        print(f"Saved {label}_histogram.png")
        print(f"  Job ID: {job.job_id()}")
        print(f"  |{TARGET_STATE}> probability: {target_prob:.4f}")
        print()

    # save combined JSON
    data = {
        'backend': IBM_BACKEND,
        'run_dir': os.path.basename(run_dir),
        'qubits': N_QUBITS,
        'target': TARGET_STATE,
        'shots': SHOTS,
        'results': [
            {'label': r.label, 'counts': r.counts, 'target_prob': r.target_prob, 'job_id': r.job_id}
            for r in results
        ],
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print(f"results.json saved")

    # comparison bar chart
    save_comparison_chart(results, run_dir)
    print(f"comparison_bars.png saved")

    # update summary across all runs
    update_summary()
    print(f"summary.json + summary_chart.png updated at {BASE_DIR}/")

    # print run summary
    print("\nSummary:")
    print("-" * 40)
    for r in sorted(results, key=lambda x: -x.target_prob):
        print(f"  {r.label}: {r.target_prob:.4f}  (job: {r.job_id})")


# ───────────────────────────
#  Cross-run summary
# ───────────────────────────

def update_summary() -> None:
    """Aggregate results from all run_XXX folders into summary.json + chart."""
    runs = sorted(
        d for d in os.listdir(BASE_DIR)
        if re.fullmatch(r'run_\d{3}', d)
    )

    all_entries: list[dict] = []
    for run in runs:
        path = os.path.join(BASE_DIR, run, 'results.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data['results']:
            all_entries.append({
                'run': run,
                'backend': data['backend'],
                'label': r['label'],
                'target_prob': r['target_prob'],
            })

    if not all_entries:
        return

    # save summary JSON
    with open(os.path.join(BASE_DIR, 'summary.json'), 'w') as f:
        json.dump(all_entries, f, indent=2)

    # group by combo label for chart
    by_label: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in all_entries:
        by_label[e['label']].append((e['run'], e['target_prob']))

    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
    plt.figure(figsize=(10, 5))

    for idx, (label, points) in enumerate(sorted(by_label.items())):
        points.sort(key=lambda x: x[0])
        xs = range(len(points))
        ys = [p[1] for p in points]
        plt.plot(xs, ys, 'o-', color=colors[idx % len(colors)], label=label)
        for x, y in zip(xs, ys):
            plt.text(x, y + 0.015, f'{y:.3f}', ha='center', fontsize=8,
                     color=colors[idx % len(colors)])

    plt.axhline(1 / 2**N_QUBITS, color='gray', linestyle='--',
                label=f'Random (1/{2**N_QUBITS})')
    plt.xticks(range(len(runs)), runs, rotation=30)
    plt.ylabel(f'Probability of |{TARGET_STATE}>')
    plt.title('Grover: Cross-Run Performance Comparison')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'summary_chart.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    main()
