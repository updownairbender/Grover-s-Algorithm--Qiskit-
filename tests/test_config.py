"""Unit tests for config values."""
import os
from src import config


class TestConfigValues:
    def test_n_qubits_is_positive_int(self):
        assert isinstance(config.N_QUBITS, int), "N_QUBITS should be int"
        assert config.N_QUBITS > 0, "N_QUBITS should be positive"
        assert config.N_QUBITS >= 2, "N_QUBITS should be >= 2 for meaningful Grover"

    def test_target_state_is_valid(self):
        assert isinstance(config.TARGET_STATE, str), "TARGET_STATE should be str"
        assert len(config.TARGET_STATE) == config.N_QUBITS, \
            f"TARGET_STATE length ({len(config.TARGET_STATE)}) should match N_QUBITS ({config.N_QUBITS})"
        assert all(b in '01' for b in config.TARGET_STATE), \
            f"TARGET_STATE '{config.TARGET_STATE}' should only contain '0' and '1'"

    def test_shots_is_positive_int(self):
        assert isinstance(config.SHOTS, int), "SHOTS should be int"
        assert config.SHOTS > 0, "SHOTS should be positive"

    def test_ibm_channel_is_string(self):
        assert isinstance(config.IBM_CHANNEL, str)

    def test_ibm_token_from_env(self):
        """IBM_TOKEN should default to empty string if env not set."""
        assert config.IBM_TOKEN == os.getenv('IBM_TOKEN', ''), "IBM_TOKEN mismatch"

    def test_ibm_instance_is_string(self):
        assert isinstance(config.IBM_INSTANCE, str)

    def test_ibm_backend_is_string(self):
        assert isinstance(config.IBM_BACKEND, str)
