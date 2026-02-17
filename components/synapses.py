"""
Synapse models: from basic connections to metabolic structural synapses.

Classes are ordered by complexity level:
  - Synapse: Simple connection with delay (Level 1)
  - PlasticSynapse: STDP + STP plasticity (Level 2)
  - MetabolicPlasticSynapse: ATP-modulated transmission and learning (Level 3)
  - StructuralSynapse: Distance-based delay with full plasticity (Level 4)
"""

import numpy as np
from collections import deque

from .base import SynapseParams, STDP_WINDOW
from .neurons import HodgkinHuxleyNeuron, MetabolicNeuron, StructuralNeuron


# ============================================================================
# LEVEL 1: BASIC SYNAPSE
# ============================================================================

class Synapse:
    """
    Simple synaptic connection with delay and temporal dynamics.

    When the presynaptic neuron fires, a current pulse arrives at the
    postsynaptic neuron after a propagation delay.
    """

    def __init__(self, pre_neuron: HodgkinHuxleyNeuron, params: SynapseParams = None):
        self.pre_neuron = pre_neuron
        self.params = params or SynapseParams()
        self.spike_queue = []
        self.pulse_end_time = -np.inf

    def on_presynaptic_spike(self, t: float):
        """Called when presynaptic neuron fires."""
        arrival_time = t + self.params.delay
        self.spike_queue.append(arrival_time)

    def get_current(self, t: float) -> float:
        """Get postsynaptic current at time t."""
        while self.spike_queue and self.spike_queue[0] <= t:
            self.spike_queue.pop(0)
            self.pulse_end_time = t + self.params.pulse_duration

        if t < self.pulse_end_time:
            return self.params.weight
        return 0.0


# ============================================================================
# LEVEL 2: PLASTIC SYNAPSE
# ============================================================================

class PlasticSynapse:
    """
    Synapse with both short-term and long-term plasticity.

    Short-Term Plasticity (STP):
        - Facilitation: repeated use temporarily strengthens the synapse
        - Depression: repeated use temporarily weakens the synapse

    Spike-Timing-Dependent Plasticity (STDP):
        - LTP: pre fires before post -> weight increases
        - LTD: pre fires after post  -> weight decreases
    """

    def __init__(self, pre_neuron: HodgkinHuxleyNeuron, params: SynapseParams = None):
        self.pre_neuron = pre_neuron
        self.params = params or SynapseParams()
        self.weight = self.params.weight

        self.spike_queue: deque = deque()
        self.pulse_end_time = -np.inf

        # STDP state
        self.last_pre_spike_time = -np.inf
        self.last_post_spike_time = -np.inf

        # STP state
        self.facilitation = 1.0
        self.depression = 1.0

    def on_presynaptic_spike(self, t: float):
        """Handle a presynaptic spike: update STP and schedule arrival."""
        dt = t - self.last_pre_spike_time if self.last_pre_spike_time > 0 else 0.0
        self.last_pre_spike_time = t
        p = self.params

        self.facilitation = 1.0 + (self.facilitation - 1.0) * np.exp(-dt / p.tau_facilitation)
        self.depression = 1.0 - (1.0 - self.depression) * np.exp(-dt / p.tau_depression)
        self.facilitation += p.U_facilitation * (1.0 - self.facilitation)
        self.depression *= (1.0 - p.U_depression)

        self.spike_queue.append(t + p.delay)

    def get_current(self, t: float) -> float:
        """Get postsynaptic current at time t, modulated by STP."""
        while self.spike_queue and self.spike_queue[0] <= t:
            self.spike_queue.popleft()
            self.pulse_end_time = t + self.params.pulse_duration

        if t < self.pulse_end_time:
            effective_weight = self.weight * self.facilitation * self.depression
            return -effective_weight if self.params.is_inhibitory else effective_weight

        return 0.0

    def apply_stdp(self, post_spike_time: float):
        """
        Apply STDP learning rule on a postsynaptic spike.

        Adjusts the base weight based on the time difference between
        pre and post spikes. Only excitatory synapses undergo STDP.
        """
        if self.params.is_inhibitory or self.last_pre_spike_time < 0:
            return

        self.last_post_spike_time = post_spike_time
        delta_t = post_spike_time - self.last_pre_spike_time
        delta_w = 0.0
        p = self.params

        if 0 < delta_t < STDP_WINDOW:
            delta_w = p.stdp_ltp_rate * np.exp(-delta_t / STDP_WINDOW)
        elif -STDP_WINDOW < delta_t < 0:
            delta_w = -p.stdp_ltd_rate * np.exp(delta_t / STDP_WINDOW)

        if delta_w != 0.0:
            self.weight = np.clip(self.weight + delta_w, p.min_weight, p.max_weight)

    def reset_dynamic_state(self):
        """Reset transient state for a new pattern, preserve learned weight."""
        self.spike_queue.clear()
        self.pulse_end_time = -np.inf
        self.last_pre_spike_time = -np.inf
        self.last_post_spike_time = -np.inf
        self.facilitation = 1.0
        self.depression = 1.0


# ============================================================================
# LEVEL 3: METABOLIC PLASTIC SYNAPSE
# ============================================================================

class MetabolicPlasticSynapse:
    """
    Synapse with STP, STDP, and energy-dependent modulation.

    Extends PlasticSynapse:
      - Synaptic transmission is scaled by presynaptic ATP
      - STDP weight updates are scaled by average ATP of pre and post neurons
    """

    def __init__(self, pre_neuron: MetabolicNeuron, params: SynapseParams = None):
        self.pre_neuron = pre_neuron
        self.params = params or SynapseParams()
        self.weight = self.params.weight

        self.spike_queue: deque = deque()
        self.pulse_end_time = -np.inf
        self.last_pre_spike_time = -np.inf
        self.last_post_spike_time = -np.inf
        self.facilitation = 1.0
        self.depression = 1.0

    def on_presynaptic_spike(self, t: float):
        dt = t - self.last_pre_spike_time if self.last_pre_spike_time > 0 else 0.0
        self.last_pre_spike_time = t
        p = self.params

        self.facilitation = 1.0 + (self.facilitation - 1.0) * np.exp(-dt / p.tau_facilitation)
        self.depression = 1.0 - (1.0 - self.depression) * np.exp(-dt / p.tau_depression)
        self.facilitation += p.U_facilitation * (1.0 - self.facilitation)
        self.depression *= (1.0 - p.U_depression)

        self.spike_queue.append(t + p.delay)

    def get_current(self, t: float) -> float:
        """Get postsynaptic current, modulated by STP and presynaptic ATP."""
        while self.spike_queue and self.spike_queue[0] <= t:
            self.spike_queue.popleft()
            self.pulse_end_time = t + self.params.pulse_duration

        if t < self.pulse_end_time:
            atp_factor = np.tanh(self.pre_neuron.metabolism.get_atp() / 4.0)
            effective_weight = self.weight * self.facilitation * self.depression * atp_factor
            return -effective_weight if self.params.is_inhibitory else effective_weight

        return 0.0

    def apply_stdp(self, post_neuron: MetabolicNeuron, post_spike_time: float):
        """Apply STDP learning rule, modulated by metabolic state."""
        if self.params.is_inhibitory or self.last_pre_spike_time < 0:
            return

        self.last_post_spike_time = post_spike_time
        delta_t = post_spike_time - self.last_pre_spike_time
        delta_w = 0.0
        p = self.params

        if 0 < delta_t < STDP_WINDOW:
            delta_w = p.stdp_ltp_rate * np.exp(-delta_t / STDP_WINDOW)
        elif -STDP_WINDOW < delta_t < 0:
            delta_w = -p.stdp_ltd_rate * np.exp(delta_t / STDP_WINDOW)

        if delta_w != 0.0:
            pre_atp = self.pre_neuron.metabolism.get_atp()
            post_atp = post_neuron.metabolism.get_atp()
            avg_atp = (pre_atp + post_atp) / 2.0
            atp_scaling = np.tanh(avg_atp / 5.0)
            delta_w *= atp_scaling
            self.weight = np.clip(self.weight + delta_w, p.min_weight, p.max_weight)

    def reset_dynamic_state(self):
        self.spike_queue.clear()
        self.pulse_end_time = -np.inf
        self.last_pre_spike_time = -np.inf
        self.last_post_spike_time = -np.inf
        self.facilitation = 1.0
        self.depression = 1.0


# ============================================================================
# LEVEL 4: STRUCTURAL SYNAPSE
# ============================================================================

class StructuralSynapse:
    """
    Synapse with STP, STDP, energy modulation, and distance-based delay.

    Identical to MetabolicPlasticSynapse but accepts a custom delay
    computed from the spatial distance between pre and post neurons.
    """

    def __init__(self, pre_neuron: StructuralNeuron, params: SynapseParams = None,
                 delay: float = 1.0):
        self.pre_neuron = pre_neuron
        self.params = params or SynapseParams()
        self.weight = self.params.weight
        self.delay = delay

        self.spike_queue: deque = deque()
        self.pulse_end_time = -np.inf
        self.last_pre_spike_time = -np.inf
        self.last_post_spike_time = -np.inf
        self.facilitation = 1.0
        self.depression = 1.0

    def on_presynaptic_spike(self, t: float):
        dt = t - self.last_pre_spike_time if self.last_pre_spike_time > 0 else 0.0
        self.last_pre_spike_time = t
        p = self.params

        self.facilitation = 1.0 + (self.facilitation - 1.0) * np.exp(-dt / p.tau_facilitation)
        self.depression = 1.0 - (1.0 - self.depression) * np.exp(-dt / p.tau_depression)
        self.facilitation += p.U_facilitation * (1.0 - self.facilitation)
        self.depression *= (1.0 - p.U_depression)

        self.spike_queue.append(t + self.delay)

    def get_current(self, t: float) -> float:
        while self.spike_queue and self.spike_queue[0] <= t:
            self.spike_queue.popleft()
            self.pulse_end_time = t + self.params.pulse_duration

        if t < self.pulse_end_time:
            atp_factor = np.tanh(self.pre_neuron.metabolism.get_atp() / 4.0)
            effective_weight = self.weight * self.facilitation * self.depression * atp_factor
            return -effective_weight if self.params.is_inhibitory else effective_weight

        return 0.0

    def apply_stdp(self, post_neuron: StructuralNeuron, post_spike_time: float):
        if self.params.is_inhibitory or self.last_pre_spike_time < 0:
            return

        self.last_post_spike_time = post_spike_time
        delta_t = post_spike_time - self.last_pre_spike_time
        delta_w = 0.0
        p = self.params

        if 0 < delta_t < STDP_WINDOW:
            delta_w = p.stdp_ltp_rate * np.exp(-delta_t / STDP_WINDOW)
        elif -STDP_WINDOW < delta_t < 0:
            delta_w = -p.stdp_ltd_rate * np.exp(delta_t / STDP_WINDOW)

        if delta_w != 0.0:
            pre_atp = self.pre_neuron.metabolism.get_atp()
            post_atp = post_neuron.metabolism.get_atp()
            avg_atp = (pre_atp + post_atp) / 2.0
            atp_scaling = np.tanh(avg_atp / 5.0)
            delta_w *= atp_scaling
            self.weight = np.clip(self.weight + delta_w, p.min_weight, p.max_weight)

    def reset_dynamic_state(self):
        self.spike_queue.clear()
        self.pulse_end_time = -np.inf
        self.last_pre_spike_time = -np.inf
        self.last_post_spike_time = -np.inf
        self.facilitation = 1.0
        self.depression = 1.0
