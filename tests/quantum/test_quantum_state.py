# tests/quantum/test_quantum_state.py
def test_chaos_application():
    qs = QuantumState(2, chaos_factor=0.1)
    initial_state = qs._state.copy()
    qs.apply_chaos()
    assert not np.allclose(initial_state.data, qs._state.data)
