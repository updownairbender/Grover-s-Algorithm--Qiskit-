"""Tests for scalability benchmark script."""
import os
import sys
import json
import subprocess


def _run_script(script_path, cwd=None, timeout=120):
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True,
        cwd=cwd or os.path.dirname(script_path),
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


class TestBenchmarkScalability:
    def test_script_runs_successfully(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'benchmark_scalability.py')
        rc, stdout, stderr = _run_script(script, cwd=project_root, timeout=600)
        assert rc == 0, f"benchmark_scalability.py failed:\n{stderr}"
        assert 'n=3' in stdout
        assert 'n=10' in stdout
        assert 'O2+D2' in stdout

    def test_output_files_created(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'benchmark_scalability.py')
        _run_script(script, cwd=project_root, timeout=600)
        out_dir = os.path.join(project_root, 'output', 'scalability_benchmark')
        assert os.path.exists(os.path.join(out_dir, 'scalability.json'))
        assert os.path.exists(os.path.join(out_dir, 'scalability_chart.png'))

    def test_json_contents(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'benchmark_scalability.py')
        _run_script(script, cwd=project_root, timeout=600)
        json_path = os.path.join(project_root, 'output', 'scalability_benchmark', 'scalability.json')
        with open(json_path) as f:
            data = json.load(f)
        assert 'qubit_range' in data
        assert 'results' in data
        assert data['qubit_range'] == [3, 4, 5, 6, 7, 8, 9, 10]

    def test_all_combos_present(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'benchmark_scalability.py')
        _run_script(script, cwd=project_root, timeout=600)
        json_path = os.path.join(project_root, 'output', 'scalability_benchmark', 'scalability.json')
        with open(json_path) as f:
            data = json.load(f)
        combos_found = set(r['combo'] for r in data['results'])
        assert combos_found == {'O1+D1', 'O1+D2', 'O2+D1', 'O2+D2'}

    def test_all_n_values_present(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'benchmark_scalability.py')
        _run_script(script, cwd=project_root, timeout=600)
        json_path = os.path.join(project_root, 'output', 'scalability_benchmark', 'scalability.json')
        with open(json_path) as f:
            data = json.load(f)
        n_values = set(r['n_qubits'] for r in data['results'])
        assert n_values == {3, 4, 5, 6, 7, 8, 9, 10}

    def test_metrics_are_positive(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'benchmark_scalability.py')
        _run_script(script, cwd=project_root, timeout=600)
        json_path = os.path.join(project_root, 'output', 'scalability_benchmark', 'scalability.json')
        with open(json_path) as f:
            data = json.load(f)
        for row in data['results']:
            assert row['raw_depth'] > 0
            assert row['raw_gates'] > 0
            assert row['transpiled_depth'] > 0
            assert row['transpiled_gates'] > 0

    def test_o2d2_is_best(self):
        """O2+D2 should have the lowest raw depth and gate count at each n."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'benchmark_scalability.py')
        _run_script(script, cwd=project_root, timeout=600)
        json_path = os.path.join(project_root, 'output', 'scalability_benchmark', 'scalability.json')
        with open(json_path) as f:
            data = json.load(f)
        for n in [3, 4, 5, 6, 7, 8, 9, 10]:
            rows = [r for r in data['results'] if r['n_qubits'] == n]
            o2d2 = next(r for r in rows if r['combo'] == 'O2+D2')
            others = [r for r in rows if r['combo'] != 'O2+D2']
            for other in others:
                assert o2d2['raw_depth'] <= other['raw_depth'], \
                    f"O2+D2 depth {o2d2['raw_depth']} > {other['combo']} {other['raw_depth']} at n={n}"
                assert o2d2['raw_gates'] <= other['raw_gates'], \
                    f"O2+D2 gates {o2d2['raw_gates']} > {other['combo']} {other['raw_gates']} at n={n}"
