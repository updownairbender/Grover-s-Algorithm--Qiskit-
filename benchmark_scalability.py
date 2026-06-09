"""Benchmark how depth and gate count scale with number of qubits (3–6)."""
import json
import os
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

from src.config import TARGET_STATE
from src.oracles import build_oracle_v1, build_oracle_v2
from src.diffuser import build_diffuser_v1, build_diffuser_v2

OUTPUT_DIR = 'output/scalability_benchmark'

COMBOS: list[tuple[str, str, str]] = [
    ('O1+D1', 'oracle_v1', 'diffuser_v1'),
    ('O1+D2', 'oracle_v1', 'diffuser_v2'),
    ('O2+D1', 'oracle_v2', 'diffuser_v1'),
    ('O2+D2', 'oracle_v2', 'diffuser_v2'),
]

ORACLES = {'oracle_v1': build_oracle_v1, 'oracle_v2': build_oracle_v2}
DIFFUSERS = {'diffuser_v1': build_diffuser_v1, 'diffuser_v2': build_diffuser_v2}

N_QUBITS_RANGE = [3, 4, 5, 6, 7, 8, 9, 10]

COLORS = {
    'O1+D1': '#4c72b0',
    'O1+D2': '#55a868',
    'O2+D1': '#c44e52',
    'O2+D2': '#8172b2',
}
STYLES = {'O1+D1': ('o', '-'), 'O1+D2': ('s', '--'), 'O2+D1': ('^', '-.'), 'O2+D2': ('D', ':')}


def build_full_circuit(n_qubits: int, target_state: str, oracle_name: str, diffuser_name: str) -> QuantumCircuit:
    oracle_fn = ORACLES[oracle_name]
    diffuser_fn = DIFFUSERS[diffuser_name]

    oracle_gate = oracle_fn(n_qubits, target_state[:n_qubits])
    diffuser_gate = diffuser_fn(n_qubits)

    qc = QuantumCircuit(n_qubits, name=f'{oracle_name}+{diffuser_name}')
    qc.h(range(n_qubits))
    qc.barrier()
    n_iterations = int(math.pi / 4 * math.sqrt(2 ** n_qubits))
    for i in range(n_iterations):
        qc.append(oracle_gate, range(n_qubits))
        qc.barrier()
        qc.append(diffuser_gate, range(n_qubits))
        if i < n_iterations - 1:
            qc.barrier()
    return qc


def _count_ops(qc: QuantumCircuit) -> dict[str, int]:
    dag = circuit_to_dag(qc)
    counts: dict[str, int] = {}
    for node in dag.op_nodes():
        name = node.op.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results: list[dict] = []

    for nq in N_QUBITS_RANGE:
        target = TARGET_STATE[:nq] if len(TARGET_STATE) >= nq else TARGET_STATE.zfill(nq)

        backend = GenericBackendV2(
            num_qubits=nq,
            coupling_map=None,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'],
            seed=42,
        )
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=1,
            seed_transpiler=42,
        )

        for label, oracle_name, diffuser_name in COMBOS:
            qc = build_full_circuit(nq, target, oracle_name, diffuser_name)
            raw_depth = qc.decompose().depth()
            raw_ops = _count_ops(qc.decompose())
            raw_gates = sum(raw_ops.values())

            qc_t = pm.run(qc)
            transpiled_depth = qc_t.depth()
            transpiled_ops = _count_ops(qc_t)
            transpiled_gates = sum(transpiled_ops.values())

            results.append({
                'n_qubits': nq,
                'combo': label,
                'oracle': oracle_name,
                'diffuser': diffuser_name,
                'raw_depth': raw_depth,
                'raw_gates': raw_gates,
                'transpiled_depth': transpiled_depth,
                'transpiled_gates': transpiled_gates,
            })

            print(f"  n={nq} {label:>6}: raw depth={raw_depth:4d} gates={raw_gates:4d}  "
                  f"-> transpiled depth={transpiled_depth:4d} gates={transpiled_gates:4d}")

    # save JSON
    with open(os.path.join(OUTPUT_DIR, 'scalability.json'), 'w') as f:
        json.dump({'qubit_range': N_QUBITS_RANGE, 'results': results}, f, indent=2)

    # build plot data
    qubit_labels = [str(n) for n in N_QUBITS_RANGE]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('raw_depth', 'Raw Circuit Depth', axes[0, 0]),
        ('raw_gates', 'Raw Gate Count', axes[0, 1]),
        ('transpiled_depth', 'Transpiled Circuit Depth', axes[1, 0]),
        ('transpiled_gates', 'Transpiled Gate Count', axes[1, 1]),
    ]

    for metric, title, ax in metrics:
        for label, _, _ in COMBOS:
            y_vals = [
                next(r[metric] for r in results if r['n_qubits'] == nq and r['combo'] == label)
                for nq in N_QUBITS_RANGE
            ]
            marker, linestyle = STYLES[label]
            ax.plot(qubit_labels, y_vals, marker=marker, linestyle=linestyle,
                    color=COLORS[label], label=label, linewidth=2, markersize=8)

        ax.set_xlabel('Number of Qubits', fontsize=12)
        ax.set_ylabel(title.split()[-1], fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # annotate values
        for label, _, _ in COMBOS:
            y_vals = [
                next(r[metric] for r in results if r['n_qubits'] == nq and r['combo'] == label)
                for nq in N_QUBITS_RANGE
            ]
            for i, (x, y) in enumerate(zip(qubit_labels, y_vals)):
                offset = 5 if metric.endswith('depth') else 8
                ax.annotate(str(y), (x, y), textcoords='offset points',
                            xytext=(0, 8 if i % 2 == 0 else -16), ha='center', fontsize=7,
                            color=COLORS[label], fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scalability_chart.png'), dpi=150)
    plt.close()
    print(f"\nChart saved to {OUTPUT_DIR}/scalability_chart.png")
    print(f"Data saved to {OUTPUT_DIR}/scalability.json")


if __name__ == '__main__':
    main()
