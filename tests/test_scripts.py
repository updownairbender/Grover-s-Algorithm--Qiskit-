"""Integration tests for runnable scripts.

Tests that each script executes without errors and produces
the expected output files.
"""
import os
import sys
import importlib.util
import tempfile


def _run_script(script_path: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Run a Python script and capture stdout, stderr, and exit code."""
    import subprocess
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=cwd or os.path.dirname(script_path),
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


class TestRunSimulation:
    def test_script_runs_successfully(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'run_simulation.py')
        rc, stdout, stderr = _run_script(script, cwd=project_root)
        assert rc == 0, f"run_simulation.py failed:\n{stderr}"
        assert 'correct' in stdout, "Expected 'correct' in output"

    def test_output_files_created(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'run_simulation.py')
        _run_script(script, cwd=project_root)
        assert os.path.exists(os.path.join(project_root, 'output', 'circuit.png'))
        assert os.path.exists(os.path.join(project_root, 'output', 'histogram.png'))


class TestCompareIterations:
    def test_script_runs_successfully(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'compare_iterations.py')
        rc, stdout, stderr = _run_script(script, cwd=project_root)
        assert rc == 0, f"compare_iterations.py failed:\n{stderr}"
        assert 'Optimal iterations in theory' in stdout

    def test_output_files_created(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'compare_iterations.py')
        _run_script(script, cwd=project_root)
        assert os.path.exists(os.path.join(project_root, 'output', 'iteration_comparison.png'))


class TestCompareCircuitDepth:
    def test_script_runs_successfully(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'compare_circuit_depth.py')
        rc, stdout, stderr = _run_script(script, cwd=project_root)
        assert rc == 0, f"compare_circuit_depth.py failed:\n{stderr}"
        assert 'O1+D1' in stdout
        assert 'O2+D2' in stdout

    def test_output_files_created(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'compare_circuit_depth.py')
        _run_script(script, cwd=project_root)
        depth_dir = os.path.join(project_root, 'output', 'depth_comparison')
        assert os.path.exists(os.path.join(depth_dir, 'depth_comparison.json'))
        assert os.path.exists(os.path.join(depth_dir, 'depth_comparison.png'))
        for combo in ['O1+D1', 'O1+D2', 'O2+D1', 'O2+D2']:
            assert os.path.exists(os.path.join(depth_dir, f'{combo}_circuit.png'))


class TestRunNoiseSimulation:
    def test_script_runs_successfully(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'run_noise_simulation.py')
        rc, stdout, stderr = _run_script(script, cwd=project_root)
        assert rc == 0, f"run_noise_simulation.py failed:\n{stderr}"
        assert 'success rate' in stdout

    def test_output_files_created(self, tmp_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = os.path.join(project_root, 'run_noise_simulation.py')
        _run_script(script, cwd=project_root)
        noise_dir = os.path.join(project_root, 'output', 'noise_simulation')
        assert os.path.exists(os.path.join(noise_dir, 'circuit.png'))
        assert os.path.exists(os.path.join(noise_dir, 'circuit_transpiled.png'))
        assert os.path.exists(os.path.join(noise_dir, 'histogram.png'))
        assert os.path.exists(os.path.join(noise_dir, 'results.txt'))
