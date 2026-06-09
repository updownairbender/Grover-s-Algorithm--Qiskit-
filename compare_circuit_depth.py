"""Compare circuit depth and gate count across all oracle/diffuser combinations."""
import json
import os
import math

import matplotlib   
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

from src.config import N_QUBITS, TARGET_STATE
from src.oracles import build_oracle_v1, build_oracle_v2
from src.diffuser import build_diffuser_v1, build_diffuser_v2

OUTPUT_DIR = 'output/depth_comparison'

COMBOS: list[tuple[str, str, str]] = [
    ('O1+D1', 'oracle_v1', 'diffuser_v1'),
    ('O1+D2', 'oracle_v1', 'diffuser_v2'),
    ('O2+D1', 'oracle_v2', 'diffuser_v1'),
    ('O2+D2', 'oracle_v2', 'diffuser_v2'),
]

ORACLES = {'oracle_v1': build_oracle_v1, 'oracle_v2': build_oracle_v2}
DIFFUSERS = {'diffuser_v1': build_diffuser_v1, 'diffuser_v2': build_diffuser_v2}


def _count_ops(qc: QuantumCircuit) -> dict[str, int]:
    dag = circuit_to_dag(qc)
    counts: dict[str, int] = {}
    for node in dag.op_nodes():
        name = node.op.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _gate_count(qc: QuantumCircuit) -> int:
    return sum(_count_ops(qc).values())


def build_full_circuit(oracle_name: str, diffuser_name: str) -> QuantumCircuit:
    oracle_fn = ORACLES[oracle_name]
    diffuser_fn = DIFFUSERS[diffuser_name]

    oracle_gate = oracle_fn(N_QUBITS, TARGET_STATE)
    diffuser_gate = diffuser_fn(N_QUBITS)

    qc = QuantumCircuit(N_QUBITS, name=f'{oracle_name}+{diffuser_name}')
    qc.h(range(N_QUBITS))
    qc.barrier()
    n_iterations = int(math.pi / 4 * math.sqrt(2 ** N_QUBITS))
    for i in range(n_iterations):
        qc.append(oracle_gate, range(N_QUBITS))
        qc.barrier()
        qc.append(diffuser_gate, range(N_QUBITS))
        if i < n_iterations - 1:
            qc.barrier()
    return qc


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    backend = GenericBackendV2(
        num_qubits=N_QUBITS,
        coupling_map=None,
        basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'],
        seed=42,
    )
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1,
        seed_transpiler=42,
    )

    rows: list[dict] = []

    for label, oracle_name, diffuser_name in COMBOS:
        qc = build_full_circuit(oracle_name, diffuser_name)
        depth = qc.decompose().depth()
        ops = _count_ops(qc.decompose())
        total_gates = sum(ops.values())
        rows.append({
            'combo': label,
            'oracle': oracle_name,
            'diffuser': diffuser_name,
            'depth': depth,
            'total_gates': total_gates,
            'gate_breakdown': dict(sorted(ops.items())),
        })

        # save transpiled circuit diagram for each combo
        qc_transpiled = pm.run(qc)
        fig_circ = qc_transpiled.draw('mpl')
        fig_circ.savefig(f'{OUTPUT_DIR}/{label}_circuit.png', bbox_inches='tight')
        plt.close(fig_circ)
        print(f"Circuit saved: {OUTPUT_DIR}/{label}_circuit.png")

    # save JSON
    with open(os.path.join(OUTPUT_DIR, 'depth_comparison.json'), 'w') as f:
        json.dump({
            'qubits': N_QUBITS,
            'target': TARGET_STATE,
            'results': rows,
        }, f, indent=2)

    # print table
    print(f"{'Combo':>8} | {'Depth':>6} | {'Gates':>6}")
    print("-" * 28)
    for r in rows:
        print(f"{r['combo']:>8} | {r['depth']:6d} | {r['total_gates']:6d}")

    print(f"\nBreakdown by gate type:")
    print(f"{'Gate':>12}", end='')
    for r in rows:
        print(f" | {r['combo']:>10}", end='')
    print()
    print("-" * 60)
    all_gates = sorted({g for r in rows for g in r['gate_breakdown']})
    for gate in all_gates:
        print(f"{gate:>12}", end='')
        for r in rows:
            cnt = r['gate_breakdown'].get(gate, 0)
            print(f" | {cnt:10d}", end='')
        print()

    # bar chart
    labels = [r['combo'] for r in rows]
    depths = [r['depth'] for r in rows]
    gates = [r['total_gates'] for r in rows]
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    bars1 = ax1.bar(labels, depths, color=colors, edgecolor='black')
    for bar, v in zip(bars1, depths):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(v), ha='center', fontsize=10)
    ax1.set_ylabel('Circuit Depth')
    ax1.set_title('Depth Comparison')
    ax1.set_ylim(0, max(depths) * 1.2)

    bars2 = ax2.bar(labels, gates, color=colors, edgecolor='black')
    for bar, v in zip(bars2, gates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(v), ha='center', fontsize=10)
    ax2.set_ylabel('Total Gate Count')
    ax2.set_title('Gate Count Comparison')
    ax2.set_ylim(0, max(gates) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'depth_comparison.png'), dpi=150)
    plt.close()
    print(f"\nChart saved to {OUTPUT_DIR}/depth_comparison.png")
    print(f"Data saved to {OUTPUT_DIR}/depth_comparison.json")


if __name__ == '__main__':
    main()
