# -*- coding: utf-8 -*-
# Copyright © 2023 Craig Huckerby
# SPDX-License-Identifier: AGPL-3.0-only
"""
Quantum entanglement management with 11D chaos dynamics
"""

import numpy as np
from typing import List, Tuple, Dict
from qiskit.quantum_info import partial_trace

class EntanglementEngine:
    """Manages quantum entanglement with multidimensional chaos integration"""
    
    def __init__(self, chaos_factor: float = 0.05):
        self.chaos_factor = chaos_factor
        self.entanglement_layers = []
        self.fidelity_cache = {}

    def create_pairwise_entanglement(self, state: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Entangle qubit pairs with chaos-modulated CZ gates
        
        Args:
            state: Input quantum statevector
            pairs: List of (control, target) qubit pairs
            
        Returns:
            Entangled statevector
        """
        self._validate_qubit_indices(pairs, state.shape)
        entangled_state = state.copy()
        
        for ctrl, tgt in pairs:
            cz_gate = self._chaotic_cz_gate(ctrl, tgt, state.shape[0])
            entangled_state = cz_gate @ entangled_state
            
        self.entanglement_layers.append(('pairwise', pairs))
        return entangled_state

    def create_cluster_state(self, state: np.ndarray, connections: Dict[int, List[int]]]) -> np.ndarray:
        """
        Create cluster state entanglement with 11D chaos adjacency
        
        Args:
            state: Input quantum statevector
            connections: Adjacency list of qubit connections
            
        Returns:
            Cluster statevector with chaos modulation
        """
        cluster_state = state.copy()
        for qubit, neighbors in connections.items():
            for neighbor in neighbors:
                cz = self._chaotic_cz_gate(qubit, neighbor, state.shape[0])
                cluster_state = cz @ cluster_state
                cluster_state = self._apply_chaos_layer(cluster_state)
                
        self.entanglement_layers.append(('cluster', connections))
        return cluster_state

    def _chaotic_cz_gate(self, ctrl: int, tgt: int, dim: int) -> np.ndarray:
        """Generate chaos-modulated controlled-Z gate matrix"""
        base_gate = np.eye(dim)
        phase_shift = -1 * (1 + self.chaos_factor * np.random.uniform(-0.1, 0.1))
        
        # Apply phase flip to |11> state
        target_state = (1 << ctrl) | (1 << tgt)
        base_gate[target_state, target_state] = phase_shift
        
        return base_gate

    def measure_entanglement_fidelity(self, state: np.ndarray, qubit_pair: Tuple[int, int]) -> float:
        """
        Calculate entanglement fidelity between qubit pair
        
        Args:
            state: Quantum statevector
            qubit_pair: (qubit_a, qubit_b) to measure
            
        Returns:
            Fidelity score between 0 (no entanglement) and 1 (max)
        """
        reduced_state = partial_trace(state, [i for i in range(state.num_qubits) 
                                           if i not in qubit_pair])
        eigenvalues = np.linalg.eigvalsh(reduced_state)
        return 2 * (1 - np.sum(eigenvalues**2))

    def _apply_chaos_layer(self, state: np.ndarray) -> np.ndarray:
        """Apply 11D vibration effects to statevector"""
        chaos_phase = np.exp(1j * self.chaos_factor * np.random.randn(state.shape[0]))
        return chaos_phase * state / np.linalg.norm(chaos_phase * state)

    def _validate_qubit_indices(self, pairs: List[Tuple[int, int]], state_dim: int):
        """Ensure qubit indices are valid for state size"""
        num_qubits = int(np.log2(state_dim))
        for ctrl, tgt in pairs:
            if ctrl >= num_qubits or tgt >= num_qubits:
                raise ValueError(f"Qubit index exceeds state size {num_qubits}")

class EntanglementFactory:
    """Factory for creating different entanglement patterns"""
    
    @staticmethod
    def create_entangler(config: Dict) -> EntanglementEngine:
        """
        Factory method for entanglement strategies
        
        Args:
            config: Dictionary with entanglement parameters
                - 'type': 'pairwise' or 'cluster'
                - 'chaos_factor': 0.0-1.0
                - 'connections': Qubit connection map
        """
        engine = EntanglementEngine(config.get('chaos_factor', 0.05))
        
        if config['type'] == 'pairwise':
            engine.create_pairwise_entanglement = config.get('pairs', [])
        elif config['type'] == 'cluster':
            engine.create_cluster_state = config.get('connections', {})
        else:
            raise ValueError(f"Unknown entanglement type: {config['type']}")
            
        return engine
