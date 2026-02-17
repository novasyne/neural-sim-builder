"""
Shared data structures, parameters, and utility functions.

All parameter dataclasses and constants used across the component library.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List


# ============================================================================
# CONSTANTS
# ============================================================================

STDP_WINDOW = 20.0  # Time window for STDP (ms)


# ============================================================================
# SPATIAL
# ============================================================================

@dataclass
class Vector3:
    """3D spatial coordinate for neuron positioning."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: 'Vector3') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


# ============================================================================
# PARAMETER DATACLASSES
# ============================================================================

@dataclass
class NeuronParams:
    """Hodgkin-Huxley neuron parameters with biological units."""
    C_m: float = 1.0          # Membrane capacitance (uF/cm^2)
    V_rest: float = -65.0     # Resting potential (mV)
    g_Na: float = 120.0       # Sodium max conductance (mS/cm^2)
    g_K: float = 36.0         # Potassium max conductance (mS/cm^2)
    g_L: float = 0.3          # Leak conductance (mS/cm^2)
    E_Na: float = 50.0        # Sodium reversal (mV)
    E_K: float = -77.0        # Potassium reversal (mV)
    E_L: float = -54.387      # Leak reversal (mV)


@dataclass
class MetabolicParams:
    """Parameters for neuronal energy metabolism."""
    blood_glucose: float = 5.0          # Systemic glucose level (mM)
    blood_ketones: float = 0.1          # Systemic ketone level (mM)
    complex_i_efficiency: float = 1.0   # Mitochondrial electron transport efficiency (0-1)
    atp_consumption_rate: float = 1e-5  # Rate constant for ATP usage by ion pumps
    initial_atp: float = 5.0            # Starting ATP concentration (mM)


@dataclass
class SynapseParams:
    """Synapse parameters including plasticity settings."""
    weight: float = 3.0           # Initial synaptic strength
    max_weight: float = 7.5       # Upper bound on weight
    min_weight: float = 0.1       # Lower bound on weight
    delay: float = 1.0            # Propagation delay (ms)
    pulse_duration: float = 2.0   # Postsynaptic current duration (ms)
    is_inhibitory: bool = False

    # STDP parameters
    stdp_ltp_rate: float = 0.05   # Learning rate for potentiation
    stdp_ltd_rate: float = 0.06   # Learning rate for depression

    # STP parameters
    tau_facilitation: float = 200.0   # Facilitation recovery time constant (ms)
    tau_depression: float = 500.0     # Depression recovery time constant (ms)
    U_facilitation: float = 0.1       # Facilitation increment per spike
    U_depression: float = 0.2         # Depression decrement per spike


@dataclass
class SimulationParams:
    """Global simulation parameters."""
    dt: float = 0.01              # Time step (ms)
    duration: float = 250.0       # Simulation time per pattern (ms)
    n_neurons: int = 50           # Number of neurons
    excitatory_ratio: float = 0.8 # Fraction of excitatory neurons
    connection_prob: float = 0.1  # Random connectivity probability
    input_current: float = 25.0   # External input current
    n_input_neurons: int = 32     # Number of neurons receiving patterns
    training_epochs: int = 10     # Number of training passes
    synaptogenesis_rate: float = 0.0  # Probability of forming a new synapse (0 = disabled)


# ============================================================================
# PATTERN UTILITIES
# ============================================================================

def create_sparse_pattern(label: str, size: int, n_active: int = 8) -> np.ndarray:
    """
    Create a sparse binary pattern seeded by the label string.

    Same label always produces the same pattern for reproducibility.

    Args:
        label: A string label (e.g. 'A', 'B', 'pattern_1')
        size: Length of the binary vector
        n_active: Number of active (1) bits

    Returns:
        Binary numpy array of shape (size,)
    """
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(label))
    rng = np.random.RandomState(seed)
    pattern = np.zeros(size, dtype=int)
    active_indices = rng.choice(size, size=min(n_active, size), replace=False)
    pattern[active_indices] = 1
    return pattern


def create_temporal_sequence(base_pattern: np.ndarray, n_steps: int = 4,
                             n_flips: int = 4, seed: int = 0) -> List[np.ndarray]:
    """
    Create a temporal sequence by flipping bits in the base pattern.

    Each step in the sequence is a small variation of the base,
    representing a spatiotemporal input that evolves over time.

    Args:
        base_pattern: Binary array to create variations from
        n_steps: Number of temporal steps
        n_flips: Number of bits to flip per step
        seed: Random seed for reproducibility

    Returns:
        List of binary arrays forming a temporal sequence
    """
    rng = np.random.RandomState(seed)
    sequence = []
    for step in range(n_steps):
        variant = base_pattern.copy()
        flip_indices = rng.choice(len(variant), size=n_flips, replace=False)
        for idx in flip_indices:
            variant[idx] = 1 - variant[idx]
        sequence.append(variant)
    return sequence
