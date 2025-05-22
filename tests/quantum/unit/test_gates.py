import pytest
import numpy as np
from core.quantum.gates import PauliX, Hadamard, CNOTGate

def test_pauli_x_chaos():
    gate = PauliX(chaos_factor=0.1)
    gate.update_vibration_phase(1.0)
    mat = gate.matrix(dimension=3)
    assert not np.allclose(mat, np.array([[0,1],[1,0]]))

def test_hadamard_superposition():
    gate = Hadamard()
    state = np.array([1, 0], dtype=complex)
    transformed = gate.matrix() @ state
    assert np.allclose(transformed, [1/np.sqrt(2), 1/np.sqrt(2)])

def test_cnot_entanglement():
    gate = CNOTGate(chaos_factor=0.05)
    state = np.array([1, 0, 0, 0], dtype=complex)  # |00>
    entangled = gate.apply(state, [0, 1])
    assert np.allclose(entangled, [1, 0, 0, 0])  # Should remain |00>
    
    state = np.array([0, 0, 1, 0], dtype=complex)  # |10>
    entangled = gate.apply(state, [0, 1])
    assert np.allclose(entangled, [0, 0, 0, 1])  # Should become |11>
