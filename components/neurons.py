"""
Neuron models: from basic Hodgkin-Huxley to metabolic structural neurons.

Classes are ordered by complexity level:
  - HodgkinHuxleyNeuron: Basic HH dynamics (Level 1-2)
  - MetabolicComponent: ATP production/consumption with supply constraints (Level 3+)
  - MetabolicNeuron: HH neuron with energy-coupled dynamics (Level 3)
  - StructuralNeuron: Metabolic neuron with spatial position and axon (Level 4)
  - PyramidalNeuron: Excitatory neuron with long myelinated axon (Level 4)
  - BasketCell: Inhibitory neuron with short unmyelinated axon (Level 4)

Literature references for key mechanisms:
  - Refractory period: ~1-2 ms absolute in CNS (Gerstner et al., Neuronal Dynamics)
  - ATP→E_K coupling: Na+/K+-ATPase impairment under low ATP shifts K+ reversal
    (Attwell & Laughlin 2001; Rae et al. 2024)
  - Michaelis-Menten glucose transport: GLUT1 Km ≈ 7 mM (Barros et al. 2005)
  - Glycogen shunt: astrocytic buffer, 3 ATP/glucose (Dienel & Rothman 2019)
"""

import numpy as np
from typing import List

from .base import NeuronParams, MetabolicParams, Vector3


# ============================================================================
# LEVEL 1-2: BASIC HODGKIN-HUXLEY NEURON
# ============================================================================

class HodgkinHuxleyNeuron:
    """
    Hodgkin-Huxley neuron model with spike history for plasticity.

    Includes:
      - Action potential generation via Na+, K+, and leak channels
      - Absolute refractory period (literature: ~1-2 ms in CNS)
      - NaN guard for numerical stability
      - Input current clamping
    """

    def __init__(self, neuron_id: int, is_excitatory: bool = True,
                 params: NeuronParams = None):
        self.id = neuron_id
        self.is_excitatory = is_excitatory
        self.params = params or NeuronParams()

        # State variables
        self.V = self.params.V_rest
        self.m = 0.05               # Na activation gate
        self.h = 0.6                # Na inactivation gate
        self.n = 0.32               # K activation gate
        self.is_spiking = False
        self.last_spike_time = -999.0
        self.spike_times: List[float] = []

    def reset_state(self):
        """Reset dynamic state for a new simulation run, keeping learned weights."""
        self.V = self.params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
        self.last_spike_time = -999.0
        self.spike_times.clear()

    def alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def step(self, dt: float, I_external: float, I_synaptic: float, t: float) -> bool:
        """
        Advance neuron by one time step.

        Returns True if the neuron spiked this step.
        """
        p = self.params

        # Enforce refractory period
        if t - self.last_spike_time < p.refractory_period:
            self.V = p.V_rest
            return False

        # Clamp external current for numerical safety
        I_ext_clamped = np.clip(I_external, -p.max_input_current, p.max_input_current)
        I_total = I_ext_clamped + I_synaptic

        I_Na = p.g_Na * (self.m ** 3) * self.h * (self.V - p.E_Na)
        I_K = p.g_K * (self.n ** 4) * (self.V - p.E_K)
        I_L = p.g_L * (self.V - p.E_L)

        dV_dt = (I_total - I_Na - I_K - I_L) / p.C_m
        self.V += dV_dt * dt

        self.m += dt * (self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m)
        self.h += dt * (self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h)
        self.n += dt * (self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n)

        # NaN guard
        if np.isnan(self.V):
            self.V = p.V_rest

        spiked = False
        if self.V > 0.0 and not self.is_spiking:
            self.is_spiking = True
            spiked = True
            self.last_spike_time = t
            self.spike_times.append(t)
        elif self.V < 0.0:
            self.is_spiking = False

        return spiked


# ============================================================================
# LEVEL 3: METABOLIC COMPONENT
# ============================================================================

class MetabolicComponent:
    """
    Models core neuroenergetic pathways within a neuron.

    ATP production from two substrate pathways:
      - Glycolysis: glucose → 2 ATP (fast, anaerobic)
      - Oxidative phosphorylation: glucose/ketones → 28 ATP (requires Complex I)

    Supply constraints (Rae et al. 2024):
      - Michaelis-Menten glucose transport via GLUT1 at BBB
      - Glycogen shunt for transient burst energy demands

    ATP consumed by Na+/K+-ATPase to restore membrane gradients.
    """

    K_GLYCOLYSIS_ATP = 2.0     # ATP yield from glycolysis per glucose
    K_OXPHOS_ATP = 28.0        # ATP yield from oxidative phosphorylation

    def __init__(self, params: MetabolicParams):
        self.params = params
        self.atp = params.initial_atp
        self.complex_i_efficiency = params.complex_i_efficiency
        self.glycogen = params.glycogen_initial if params.glycogen_enabled else 0.0

        # Track ATP consumption for glycogen mobilization decisions
        self._recent_consumption = 0.0
        self._baseline_consumption = 0.0
        self._consumption_samples = 0

    def update(self, dt: float, atp_consumption: float,
               blood_glucose: float, blood_ketones: float):
        """
        Update ATP based on production and consumption.

        Uses Michaelis-Menten kinetics for glucose transport (GLUT1).
        Mobilizes glycogen during high-demand periods.
        """
        mp = self.params

        # Michaelis-Menten glucose transport (supply-limited)
        glucose_flux = mp.glucose_vmax * blood_glucose / (mp.glucose_km + blood_glucose)
        ketone_flux = mp.glucose_vmax * blood_ketones / (mp.glucose_km + blood_ketones)

        # Standard ATP production
        atp_generated = (
            glucose_flux * self.K_GLYCOLYSIS_ATP +
            (glucose_flux + ketone_flux) * self.K_OXPHOS_ATP * self.complex_i_efficiency
        )

        # Glycogen shunt: mobilize during high activity, replenish at rest
        glycogen_atp = 0.0
        if mp.glycogen_enabled:
            # Track consumption baseline (running average)
            self._consumption_samples += 1
            self._baseline_consumption += (atp_consumption - self._baseline_consumption) / min(
                self._consumption_samples, 10000)

            high_demand = atp_consumption > self._baseline_consumption * 1.5
            if high_demand and self.glycogen > 0.01:
                glycogen_used = min(self.glycogen, mp.glycogen_mobilization_rate * dt)
                glycogen_atp = glycogen_used * mp.glycogen_atp_yield
                self.glycogen -= glycogen_used
            elif not high_demand and self.glycogen < mp.glycogen_max:
                # Replenish glycogen from glucose during low activity
                replenish = mp.glycogen_synthesis_rate * dt
                self.glycogen = min(self.glycogen + replenish, mp.glycogen_max)

        self.atp += (atp_generated + glycogen_atp) * dt - atp_consumption
        # ATP is homeostatically regulated; in vivo [ATP] is remarkably stable
        # at ~3 mM (Zhu et al. 2012). Cap at 2x initial to prevent unbounded growth
        # while still allowing transient fluctuations.
        max_atp = mp.initial_atp * 2.0
        self.atp = np.clip(self.atp, 0.01, max_atp)

    def get_atp(self) -> float:
        return self.atp

    def get_glycogen(self) -> float:
        return self.glycogen


# ============================================================================
# LEVEL 3: METABOLIC NEURON
# ============================================================================

class MetabolicNeuron:
    """
    Hodgkin-Huxley neuron with integrated bioenergetics.

    The key metabolic coupling is the K+ reversal potential: when ATP is low,
    Na+/K+-ATPase cannot fully restore ion gradients, so E_K shifts toward
    V_rest. This is a real biophysical consequence of ATP depletion
    (Attwell & Laughlin 2001; Rae et al. 2024).

    Formula: effective_E_K = E_K * pump_eff + V_rest * (1 - pump_eff)
    where pump_eff = tanh(ATP / atp_pump_half)
    """

    def __init__(self, neuron_id: int, is_excitatory: bool = True,
                 neuron_params: NeuronParams = None,
                 metabolic_params: MetabolicParams = None):
        self.id = neuron_id
        self.is_excitatory = is_excitatory
        self.params = neuron_params or NeuronParams()
        self.metabolic_params = metabolic_params or MetabolicParams()
        self.metabolism = MetabolicComponent(self.metabolic_params)

        # Membrane state
        self.V = self.params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
        self.last_spike_time = -999.0
        self.spike_times: List[float] = []

    def reset_state(self):
        """Reset dynamic state for a new simulation run."""
        self.V = self.params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
        self.last_spike_time = -999.0
        self.spike_times.clear()

    def alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def step(self, dt: float, I_external: float, I_synaptic: float, t: float) -> bool:
        """
        Advance neuron with energy-coupled dynamics.
        Returns True if the neuron spiked this step.
        """
        p = self.params
        mp = self.metabolic_params

        # Enforce refractory period
        if t - self.last_spike_time < p.refractory_period:
            self.V = p.V_rest
            return False

        # Clamp external current
        I_ext_clamped = np.clip(I_external, -p.max_input_current, p.max_input_current)
        I_total = I_ext_clamped + I_synaptic

        I_Na = p.g_Na * (self.m ** 3) * self.h * (self.V - p.E_Na)
        I_K_ideal = p.g_K * (self.n ** 4) * (self.V - p.E_K)
        I_L = p.g_L * (self.V - p.E_L)

        # ATP consumption proportional to ionic current magnitudes
        atp_consumed = (abs(I_Na) + abs(I_K_ideal)) * mp.atp_consumption_rate * dt
        self.metabolism.update(dt, atp_consumed, mp.blood_glucose, mp.blood_ketones)

        # ATP-dependent ion pump efficiency (Na+/K+-ATPase)
        # At 3 mM: tanh(3/1.5) = tanh(2.0) ≈ 0.964 (near-normal)
        # At 1.5 mM: tanh(1.0) ≈ 0.762 (moderate impairment)
        # At 0.5 mM: tanh(0.33) ≈ 0.319 (severe, near-ischemic)
        atp_factor = np.tanh(self.metabolism.get_atp() / mp.atp_pump_half)
        effective_E_K = p.E_K * atp_factor + p.V_rest * (1.0 - atp_factor)
        I_K_effective = p.g_K * (self.n ** 4) * (self.V - effective_E_K)

        dV_dt = (I_total - (I_Na + I_K_effective + I_L)) / p.C_m
        self.V += dV_dt * dt

        self.m += dt * (self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m)
        self.h += dt * (self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h)
        self.n += dt * (self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n)

        # NaN guard
        if np.isnan(self.V):
            self.V = p.V_rest

        spiked = False
        if self.V > 0.0 and not self.is_spiking:
            self.is_spiking = True
            spiked = True
            self.last_spike_time = t
            self.spike_times.append(t)
        elif self.V < 0.0:
            self.is_spiking = False

        return spiked


# ============================================================================
# LEVEL 4: STRUCTURAL NEURONS
# ============================================================================

class StructuralNeuron:
    """
    Base neuron with spatial position, layer assignment, and metabolism.

    Subclassed by PyramidalNeuron and BasketCell to provide distinct
    axon properties and conduction characteristics.
    """

    def __init__(self, neuron_id: int, is_excitatory: bool, layer: int,
                 position: Vector3, neuron_params: NeuronParams = None,
                 metabolic_params: MetabolicParams = None,
                 axon_length: float = 5000.0, is_myelinated: bool = True):
        self.id = neuron_id
        self.is_excitatory = is_excitatory
        self.layer = layer
        self.position = position
        self.params = neuron_params or NeuronParams()
        self.metabolic_params = metabolic_params or MetabolicParams()
        self.metabolism = MetabolicComponent(self.metabolic_params)

        # Axon properties
        self.axon_length = axon_length
        self.is_myelinated = is_myelinated
        self.conduction_velocity = 15000.0 if is_myelinated else 1000.0

        # Membrane state
        self.V = self.params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
        self.last_spike_time = -999.0
        self.spike_times: List[float] = []

    def get_conduction_delay(self) -> float:
        """Base delay from axon length and conduction velocity."""
        return self.axon_length / self.conduction_velocity

    def get_delay_to(self, target: 'StructuralNeuron') -> float:
        """Total propagation delay: conduction + distance-based component."""
        distance = self.position.distance_to(target.position)
        return self.get_conduction_delay() + distance / 5000.0

    def reset_state(self):
        self.V = self.params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
        self.last_spike_time = -999.0
        self.spike_times.clear()

    def alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def step(self, dt: float, I_external: float, I_synaptic: float, t: float) -> bool:
        """Advance neuron with energy-coupled dynamics."""
        p = self.params
        mp = self.metabolic_params

        # Enforce refractory period
        if t - self.last_spike_time < p.refractory_period:
            self.V = p.V_rest
            return False

        # Clamp external current
        I_ext_clamped = np.clip(I_external, -p.max_input_current, p.max_input_current)
        I_total = I_ext_clamped + I_synaptic

        I_Na = p.g_Na * (self.m ** 3) * self.h * (self.V - p.E_Na)
        I_K_ideal = p.g_K * (self.n ** 4) * (self.V - p.E_K)
        I_L = p.g_L * (self.V - p.E_L)

        atp_consumed = (abs(I_Na) + abs(I_K_ideal)) * mp.atp_consumption_rate * dt
        self.metabolism.update(dt, atp_consumed, mp.blood_glucose, mp.blood_ketones)

        atp_factor = np.tanh(self.metabolism.get_atp() / mp.atp_pump_half)
        effective_E_K = p.E_K * atp_factor + p.V_rest * (1.0 - atp_factor)
        I_K_effective = p.g_K * (self.n ** 4) * (self.V - effective_E_K)

        dV_dt = (I_total - (I_Na + I_K_effective + I_L)) / p.C_m
        self.V += dV_dt * dt

        self.m += dt * (self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m)
        self.h += dt * (self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h)
        self.n += dt * (self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n)

        # NaN guard
        if np.isnan(self.V):
            self.V = p.V_rest

        spiked = False
        if self.V > 0.0 and not self.is_spiking:
            self.is_spiking = True
            spiked = True
            self.last_spike_time = t
            self.spike_times.append(t)
        elif self.V < 0.0:
            self.is_spiking = False

        return spiked


class PyramidalNeuron(StructuralNeuron):
    """
    Excitatory pyramidal neuron with long myelinated axon.

    Properties:
      - Excitatory (glutamatergic)
      - Long axon (8000 µm) for long-range projections
      - Myelinated for fast conduction (15 m/s)
    """

    def __init__(self, neuron_id: int, layer: int, position: Vector3,
                 neuron_params: NeuronParams = None,
                 metabolic_params: MetabolicParams = None):
        super().__init__(
            neuron_id=neuron_id,
            is_excitatory=True,
            layer=layer,
            position=position,
            neuron_params=neuron_params,
            metabolic_params=metabolic_params,
            axon_length=8000.0,
            is_myelinated=True,
        )


class BasketCell(StructuralNeuron):
    """
    Inhibitory basket cell with short unmyelinated axon.

    Properties:
      - Inhibitory (GABAergic)
      - Short axon (2000 µm) for local inhibition
      - Unmyelinated, slower conduction (1 m/s)
    """

    def __init__(self, neuron_id: int, layer: int, position: Vector3,
                 neuron_params: NeuronParams = None,
                 metabolic_params: MetabolicParams = None):
        super().__init__(
            neuron_id=neuron_id,
            is_excitatory=False,
            layer=layer,
            position=position,
            neuron_params=neuron_params,
            metabolic_params=metabolic_params,
            axon_length=2000.0,
            is_myelinated=False,
        )
