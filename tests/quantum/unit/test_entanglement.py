def test_pairwise_entanglement():
    engine = EntanglementEngine()
    state = np.array([1, 0, 0, 0], dtype=complex)  # |00>
    entangled = engine.create_pairwise_entanglement(state, [(0,1)])
    assert not np.allclose(state, entangled)
