"""
Shared data structures, parameters, and utility functions.

All parameter dataclasses and constants used across the component library.

Parameter values are sourced from established neuroscience literature:
  - Hodgkin & Huxley (1952): Ion channel dynamics (squid giant axon)
  - Bi & Poo (1998): STDP time windows
  - Song et al. (2000), Kempter et al. (1999): STDP learning rates
  - Attwell & Laughlin (2001), Howarth et al. (2012): Brain energy budget
  - Zhu et al. (2012): Brain ATP concentration (~3 mM via 31P MRS)
  - Rae et al. (2024): Brain energy constraints and allostatic load
  - Markram et al. (2015): Cortical microcircuit connectivity
  - Turrigiano & Nelson (2004): Homeostatic plasticity / AGC
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List


# ============================================================================
# CONSTANTS
# ============================================================================

STDP_WINDOW = 20.0  # Time window for STDP (ms) — Bi & Poo 1998


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
    """
    Hodgkin-Huxley neuron parameters with biological units.

    Values from Hodgkin & Huxley (1952), standard for computational models.
    """
    C_m: float = 1.0          # Membrane capacitance (uF/cm^2)
    V_rest: float = -65.0     # Resting potential (mV)
    g_Na: float = 120.0       # Sodium max conductance (mS/cm^2)
    g_K: float = 36.0         # Potassium max conductance (mS/cm^2)
    g_L: float = 0.3          # Leak conductance (mS/cm^2)
    E_Na: float = 50.0        # Sodium reversal (mV)
    E_K: float = -77.0        # Potassium reversal (mV)
    E_L: float = -54.387      # Leak reversal (mV)

    # Absolute refractory period: ~1-2 ms in CNS neurons
    # (Gerstner et al., Neuronal Dynamics; PhysiologyWeb).
    # The HH gating variables produce intrinsic relative refractoriness.
    refractory_period: float = 2.0  # ms

    # Input current safety clamp to prevent numerical instability
    max_input_current: float = 100.0  # pA


@dataclass
class MetabolicParams:
    """
    Parameters for neuronal energy metabolism.

    ATP concentration: ~3 mM (31P MRS; Zhu et al. 2012; Du et al. 2008).
    Glucose transport: Michaelis-Menten via GLUT1, Km ≈ 7 mM (Rae et al. 2024).
    Glycogen shunt: astrocytic buffer for burst activity (Dienel & Rothman 2019).
    """
    blood_glucose: float = 5.0          # Systemic glucose (mM, euglycemia)
    blood_ketones: float = 0.1          # Systemic ketones (mM, fed state)
    complex_i_efficiency: float = 1.0   # Mitochondrial Complex I efficiency (0-1)
    atp_consumption_rate: float = 1e-4  # Rate constant for ATP usage by ion pumps
    initial_atp: float = 3.0            # Starting ATP (mM) — 31P MRS measured

    # Michaelis-Menten glucose transport (BBB GLUT1 kinetics)
    # Km ≈ 7 mM for GLUT1 (Barros et al. 2005). Vmax calibrated so that
    # at steady state with typical spiking activity, ATP ≈ 3 mM.
    # Production must balance ion pump consumption per timestep.
    glucose_km: float = 7.0             # GLUT1 half-saturation (mM)
    glucose_vmax: float = 0.12          # Max glucose transport rate

    # Glycogen shunt (astrocytic energy buffer; Rae et al. 2024)
    glycogen_enabled: bool = True
    glycogen_initial: float = 5.0       # Initial glycogen (mM glucose equiv.)
    glycogen_max: float = 5.0           # Max glycogen capacity
    glycogen_mobilization_rate: float = 0.1   # Breakdown rate under demand
    glycogen_synthesis_rate: float = 0.02     # Replenishment rate at rest
    glycogen_atp_yield: float = 3.0           # ATP/glucose from glycogen (vs 2)

    # Ion pump coupling: tanh(ATP / atp_pump_half)
    # At 3 mM: tanh(2.0) ≈ 0.964 (near-normal pump function)
    atp_pump_half: float = 1.5          # Pump half-efficiency ATP level


@dataclass
class SynapseParams:
    """
    Synapse parameters including plasticity settings.

    Weight ratio: inhibitory ~4x excitatory (Markram et al. 2015).
    Max weight = 2.5x initial (bounded plasticity).
    STDP: ~0.01 LTP, ~0.012 LTD (Song et al. 2000; Kempter et al. 1999).
    """
    weight: float = 1.0           # Initial excitatory synaptic strength
    max_weight: float = 2.5       # Upper bound (2.5x initial)
    min_weight: float = 0.1       # Lower bound
    delay: float = 1.0            # Propagation delay (ms)
    pulse_duration: float = 2.0   # Postsynaptic current duration (ms)
    is_inhibitory: bool = False

    # STDP parameters — Song et al. 2000; Kempter et al. 1999
    # LTD slightly > LTP prevents runaway potentiation
    stdp_ltp_rate: float = 0.01   # LTP learning rate per spike pair
    stdp_ltd_rate: float = 0.012  # LTD learning rate per spike pair

    # STP parameters — Markram et al. 1998
    tau_facilitation: float = 200.0   # Facilitation recovery (ms)
    tau_depression: float = 500.0     # Depression recovery (ms)
    U_facilitation: float = 0.1       # Facilitation increment
    U_depression: float = 0.2         # Depression decrement


@dataclass
class AGCParams:
    """
    Automatic Gain Control (homeostatic firing-rate regulation).

    Biological basis: Turrigiano & Nelson (2004). In biology operates on
    hours-days; computational models use faster timescales.

    ATP gating prevents AGC from masking metabolic failure:
      metabolic_gate(ATP) = 1 / (1 + exp(-(ATP - atp_half) / atp_slope))
      effective_lambda = lambda_base * metabolic_gate(mean_ATP)
    """
    enabled: bool = True
    target_rate_hz: float = 35.0      # Target firing rate (Hz)
    lambda_base: float = 0.002        # Base gain adjustment rate
    interval_ms: float = 100.0        # AGC update interval (ms)
    gain_min: float = 5.0             # Minimum gain
    gain_max: float = 100.0           # Maximum gain
    initial_gain: float = 20.0        # Starting gain

    # ATP gating sigmoid
    atp_half: float = 1.5             # 50% AGC effectiveness ATP level (mM)
    atp_slope: float = 0.5            # Sigmoid slope


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
    synaptogenesis_rate: float = 0.0  # Probability of new synapse (0 = disabled)


# ============================================================================
# PATTERN UTILITIES
# ============================================================================

def create_sparse_pattern(label: str, size: int, n_active: int = 8) -> np.ndarray:
    """
    Create a sparse binary pattern seeded by the label string.
    Same label always produces the same pattern for reproducibility.
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
