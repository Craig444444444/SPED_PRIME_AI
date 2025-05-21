import pytest
from qiskit.providers.fake_provider import FakeVigo
from core.quantum.resilience import QuantumResilienceEngine

@pytest.fixture
def fake_backend():
    return FakeVigo()

def test_mitigation_calibration(fake_backend):
    engine = QuantumResilienceEngine(fake_backend)
    assert fake_backend.name in engine.calibration_data
    assert engine.get_mitigation_matrix() is not None

def test_error_rate_calculation(fake_backend):
    engine = QuantumResilienceEngine(fake_backend)
    error_rate = engine._calculate_readout_error()
    assert 0 <= error_rate <= 1
