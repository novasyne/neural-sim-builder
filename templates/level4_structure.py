"""
Level 4: Structural Cortical Networks
Builds on Level 3 by adding layered architecture, cell type diversity,
spatial organisation, and structural plasticity (synaptogenesis).

The homogeneous network is replaced with a cortical-like architecture:
  - Distinct cell types: PyramidalNeuron (excitatory) and BasketCell (inhibitory)
  - Layered organisation: Input -> Processing -> Integration layers
  - Distance-dependent connectivity with local vs long-range rules
  - Structural plasticity: the network can physically rewire itself by
    forming new synapses based on Hebbian co-activity
  - Spatiotemporal patterns that evolve over time

New concepts over Level 3:
  - PyramidalNeuron / BasketCell with distinct axon properties
  - 3D spatial positioning and distance-based delays
  - Layered network builder with per-layer connectivity rules
  - Synaptogenesis: Hebbian formation of new synapses during training
  - Spatiotemporal pattern sequences
  - Synapse count tracking over training
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import math


# ============================================================================
# PARAMETERS
# ============================================================================

STDP_WINDOW = 20.0  # Time window for STDP (ms)


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


@dataclass
class NeuronParams:
    """Hodgkin-Huxley neuron parameters."""
    C_m: float = 1.0
    V_rest: float = -65.0
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.387


@dataclass
class MetabolicParams:
    """Parameters for neuronal energy metabolism."""
    blood_glucose: float = 5.0
    blood_ketones: float = 0.1
    complex_i_efficiency: float = 1.0
    atp_consumption_rate: float = 1e-5
    initial_atp: float = 5.0


@dataclass
class SynapseParams:
    """Synapse parameters including plasticity settings."""
    weight: float = 3.0
    max_weight: float = 7.5
    min_weight: float = 0.1
    delay: float = 1.0
    pulse_duration: float = 2.0
    is_inhibitory: bool = False

    # STDP
    stdp_ltp_rate: float = 0.05
    stdp_ltd_rate: float = 0.06

    # STP
    tau_facilitation: float = 200.0
    tau_depression: float = 500.0
    U_facilitation: float = 0.1
    U_depression: float = 0.2


@dataclass
class SimulationParams:
    """Global simulation parameters."""
    dt: float = 0.01
    duration: float = 250.0           # Time per pattern step (ms)
    n_neurons: int = 256              # Total neurons across all layers
    n_input_neurons: int = 128        # Neurons in the input layer
    excitatory_ratio: float = 0.8
    input_current: float = 25.0
    training_epochs: int = 5
    synaptogenesis_rate: float = 0.0005  # Probability of forming a new synapse


# ============================================================================
# METABOLIC COMPONENT (from Level 3)
# ============================================================================

class MetabolicComponent:
    """Models neuroenergetic pathways: glycolysis and oxidative phosphorylation."""

    K_GLYCOLYSIS_ATP = 2.0
    K_OXPHOS_ATP = 28.0
    K_SUBSTRATE_UPTAKE = 0.01

    def __init__(self, params: MetabolicParams):
        self.atp = params.initial_atp
        self.complex_i_efficiency = params.complex_i_efficiency

    def update(self, dt: float, atp_consumption: float,
               blood_glucose: float, blood_ketones: float):
        glucose_flux = self.K_SUBSTRATE_UPTAKE * blood_glucose
        ketone_flux = self.K_SUBSTRATE_UPTAKE * blood_ketones
        atp_generated = (
            glucose_flux * self.K_GLYCOLYSIS_ATP +
            (glucose_flux + ketone_flux) * self.K_OXPHOS_ATP * self.complex_i_efficiency
        )
        self.atp += atp_generated * dt - atp_consumption
        self.atp = max(0.01, self.atp)

    def get_atp(self) -> float:
        return self.atp


# ============================================================================
# NEURON TYPES (New in Level 4)
# ============================================================================

class StructuralNeuron:
    """
    Base neuron with spatial position, layer assignment, and metabolism.

    Subclassed by PyramidalNeuron and BasketCell to provide distinct
    axon properties and conduction characteristics.
    """

    def __init__(self, neuron_id: int, is_excitatory: bool, layer: int,
                 position: Vector3, neuron_params: NeuronParams,
                 metabolic_params: MetabolicParams,
                 axon_length: float = 5000.0, is_myelinated: bool = True):
        self.id = neuron_id
        self.is_excitatory = is_excitatory
        self.layer = layer
        self.position = position
        self.params = neuron_params
        self.metabolic_params = metabolic_params
        self.metabolism = MetabolicComponent(metabolic_params)

        # Axon properties (determine conduction delay)
        self.axon_length = axon_length
        self.is_myelinated = is_myelinated
        # Myelinated axons conduct ~15x faster
        self.conduction_velocity = 15000.0 if is_myelinated else 1000.0

        # Membrane state
        self.V = neuron_params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
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
        """Advance neuron with energy-coupled dynamics (from Level 3)."""
        p = self.params
        mp = self.metabolic_params
        I_total = I_external + I_synaptic

        I_Na = p.g_Na * (self.m ** 3) * self.h * (self.V - p.E_Na)
        I_K_ideal = p.g_K * (self.n ** 4) * (self.V - p.E_K)
        I_L = p.g_L * (self.V - p.E_L)

        atp_consumed = (abs(I_Na) + abs(I_K_ideal)) * mp.atp_consumption_rate * dt
        self.metabolism.update(dt, atp_consumed, mp.blood_glucose, mp.blood_ketones)

        atp_factor = np.tanh(self.metabolism.get_atp() / 2.0)
        effective_E_K = p.E_K * atp_factor + p.V_rest * (1.0 - atp_factor)
        I_K_effective = p.g_K * (self.n ** 4) * (self.V - effective_E_K)

        dV_dt = (I_total - (I_Na + I_K_effective + I_L)) / p.C_m
        self.V += dV_dt * dt

        self.m += dt * (self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m)
        self.h += dt * (self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h)
        self.n += dt * (self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n)

        spiked = False
        if self.V > 0.0 and not self.is_spiking:
            self.is_spiking = True
            spiked = True
            self.spike_times.append(t)
        elif self.V < 0.0:
            self.is_spiking = False

        return spiked


class PyramidalNeuron(StructuralNeuron):
    """
    Excitatory pyramidal neuron with long myelinated axon.

    Properties:
      - Excitatory (glutamatergic)
      - Long axon (8000 um) for long-range projections
      - Myelinated for fast conduction (15 m/s)
    """

    def __init__(self, neuron_id: int, layer: int, position: Vector3,
                 neuron_params: NeuronParams, metabolic_params: MetabolicParams):
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
      - Short axon (2000 um) for local inhibition
      - Unmyelinated, slower conduction (1 m/s)
      - Targets cell bodies (somatic inhibition)
    """

    def __init__(self, neuron_id: int, layer: int, position: Vector3,
                 neuron_params: NeuronParams, metabolic_params: MetabolicParams):
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


# ============================================================================
# SYNAPSE (from Level 3, with distance-based delay)
# ============================================================================

class StructuralSynapse:
    """
    Synapse with STP, STDP, energy-dependent modulation, and distance-based delay.

    Identical to Level 3 MetabolicPlasticSynapse but accepts a custom delay
    computed from the spatial distance between pre and post neurons.
    """

    def __init__(self, pre_neuron: StructuralNeuron, params: SynapseParams,
                 delay: float):
        self.pre_neuron = pre_neuron
        self.params = params
        self.weight = params.weight
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


# ============================================================================
# PATTERN UTILITIES
# ============================================================================

def create_sparse_pattern(label: str, size: int, n_active: int = 16) -> np.ndarray:
    """Create a sparse binary pattern seeded by the label string."""
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


# ============================================================================
# LAYERED NETWORK (New in Level 4)
# ============================================================================

class LayeredNetwork:
    """
    Cortical-like network with layered architecture, cell type diversity,
    spatial organisation, and structural plasticity.

    Architecture:
      - Layer 0 (Input): All pyramidal neurons, grid-positioned
      - Layer 1 (Processing): Mixed pyramidal + basket cells
      - Layer 2 (Integration): Mixed pyramidal + basket cells

    Connectivity rules:
      - L0 -> L1: Feedforward, distance-modulated
      - L1 <-> L1: Local recurrent, strong within 150 um
      - L1 <-> L2: Sparse long-range connections
      - Same-layer connections decay with distance

    Structural plasticity:
      - New synapses form via Hebbian synaptogenesis when pre and post
        neurons fire within the STDP window
    """

    def __init__(self, sim_params: SimulationParams, metabolic_params: MetabolicParams):
        self.params = sim_params
        self.metabolic_params = metabolic_params
        neuron_params = NeuronParams()

        self.neurons: List[StructuralNeuron] = []
        self.synapses: Dict[int, List[StructuralSynapse]] = {}

        # Recording
        self.spike_history: List[Tuple[float, int]] = []
        self.weight_snapshots: List[float] = []
        self.atp_snapshots: List[float] = []
        self.synapse_count_snapshots: List[int] = []
        self.last_spike_times: Dict[int, float] = {}

        # Build layered architecture
        self._create_neurons(neuron_params)
        self._create_layered_connections()

    def _create_neurons(self, neuron_params: NeuronParams):
        """Create neurons across layers with appropriate cell types."""
        mp = self.metabolic_params
        n_input = self.params.n_input_neurons
        n_processing = (self.params.n_neurons - n_input) // 2
        n_integration = self.params.n_neurons - n_input - n_processing
        neuron_id = 0

        # Layer 0: Input layer - all pyramidal, grid positions
        for i in range(n_input):
            col = i % 16
            row = i // 16
            pos = Vector3((col - 8) * 20.0, (row - 4) * 20.0, 0.0)
            self.neurons.append(
                PyramidalNeuron(neuron_id, 0, pos, neuron_params, mp)
            )
            neuron_id += 1

        # Layer 1 and 2: Processing layers - mixed cell types
        for layer, n_in_layer in [(1, n_processing), (2, n_integration)]:
            n_exc = int(n_in_layer * self.params.excitatory_ratio)
            for i in range(n_in_layer):
                pos = Vector3(
                    np.random.uniform(-400, 400),
                    np.random.uniform(-400, 400),
                    layer * 200.0,
                )
                if i < n_exc:
                    self.neurons.append(
                        PyramidalNeuron(neuron_id, layer, pos, neuron_params, mp)
                    )
                else:
                    self.neurons.append(
                        BasketCell(neuron_id, layer, pos, neuron_params, mp)
                    )
                neuron_id += 1

        # Initialise synapse dict
        for n in self.neurons:
            self.synapses[n.id] = []

    def _create_layered_connections(self):
        """Create distance-dependent connectivity respecting layer rules."""
        LOCAL_PROB = 0.4
        DISTANT_PROB = 0.1

        exc_params = SynapseParams(weight=3.0, max_weight=7.5, is_inhibitory=False)
        inh_params = SynapseParams(weight=5.0, max_weight=12.5, is_inhibitory=True)

        for pre in self.neurons:
            for post in self.neurons:
                if pre.id == post.id:
                    continue

                distance = pre.position.distance_to(post.position)
                connection_prob = 0.0

                # L0 -> L1: Feedforward, distance-modulated
                if pre.layer == 0 and post.layer == 1:
                    connection_prob = LOCAL_PROB * np.exp(-distance / 250.0)

                # Within processing layers or between L1 <-> L2
                elif pre.layer > 0 and post.layer > 0:
                    if distance < 150.0:
                        connection_prob = LOCAL_PROB
                    else:
                        connection_prob = DISTANT_PROB
                    connection_prob *= np.exp(-distance / 400.0)

                if np.random.random() < connection_prob:
                    params = exc_params if pre.is_excitatory else inh_params
                    delay = pre.get_delay_to(post)
                    syn = StructuralSynapse(pre, params, delay)
                    self.synapses[post.id].append(syn)

    def _check_connection_exists(self, pre_id: int, post_id: int) -> bool:
        """Check if a synapse from pre to post already exists."""
        for syn in self.synapses[post_id]:
            if syn.pre_neuron.id == pre_id:
                return True
        return False

    def _attempt_synaptogenesis(self, post_idx: int, t: float):
        """
        Attempt to form new synapses via Hebbian synaptogenesis.

        When a post-synaptic neuron fires, check if any pre-synaptic neurons
        fired recently (within STDP window). If so, with small probability,
        form a new synapse. Only processing-layer neurons can receive new synapses.
        """
        post_neuron = self.neurons[post_idx]
        if post_neuron.layer == 0:
            return  # No new synapses onto input layer

        exc_params = SynapseParams(weight=0.2, max_weight=7.5, is_inhibitory=False)
        inh_params = SynapseParams(weight=0.3, max_weight=12.5, is_inhibitory=True)

        for pre_idx, pre_neuron in enumerate(self.neurons):
            if pre_idx == post_idx:
                continue

            pre_last_spike = self.last_spike_times.get(pre_idx, -np.inf)
            delta_t = t - pre_last_spike

            if 0 < delta_t < STDP_WINDOW:
                if np.random.random() < self.params.synaptogenesis_rate:
                    if not self._check_connection_exists(pre_idx, post_idx):
                        params = exc_params if pre_neuron.is_excitatory else inh_params
                        delay = pre_neuron.get_delay_to(post_neuron)
                        new_syn = StructuralSynapse(pre_neuron, params, delay)
                        self.synapses[post_idx].append(new_syn)

    def get_mean_excitatory_weight(self) -> float:
        weights = []
        for syn_list in self.synapses.values():
            for syn in syn_list:
                if not syn.params.is_inhibitory:
                    weights.append(syn.weight)
        return float(np.mean(weights)) if weights else 0.0

    def get_mean_atp(self) -> float:
        return float(np.mean([n.metabolism.get_atp() for n in self.neurons]))

    def get_synapse_count(self) -> int:
        return sum(len(s) for s in self.synapses.values())

    def reset_neuron_states(self):
        for neuron in self.neurons:
            neuron.reset_state()
        for syn_list in self.synapses.values():
            for syn in syn_list:
                syn.reset_dynamic_state()

    def _run_pattern_sequence(self, sequence: List[np.ndarray],
                              enable_synaptogenesis: bool = True):
        """
        Run a temporal pattern sequence through the network.

        Each step in the sequence is presented for self.params.duration ms.
        """
        dt = self.params.dt
        step_duration = self.params.duration
        total_time = step_duration * len(sequence)
        n_total_steps = int(total_time / dt)

        for sim_step in range(n_total_steps):
            t = sim_step * dt

            # Which temporal pattern step are we in?
            pattern_idx = min(int(t / step_duration), len(sequence) - 1)
            current_pattern = sequence[pattern_idx]

            spiked_neurons: List[int] = []

            for neuron in self.neurons:
                I_ext = 0.0
                if neuron.id < self.params.n_input_neurons:
                    if neuron.id < len(current_pattern) and current_pattern[neuron.id] == 1:
                        I_ext = self.params.input_current

                I_syn = sum(syn.get_current(t) for syn in self.synapses[neuron.id])
                spiked = neuron.step(dt, I_ext, I_syn, t)

                if spiked:
                    spiked_neurons.append(neuron.id)
                    self.spike_history.append((t, neuron.id))
                    self.last_spike_times[neuron.id] = t

            # Propagate spikes, apply STDP, attempt synaptogenesis
            for nid in spiked_neurons:
                spiking_neuron = self.neurons[nid]

                for post_id, syn_list in self.synapses.items():
                    for syn in syn_list:
                        if syn.pre_neuron.id == nid:
                            syn.on_presynaptic_spike(t)

                for syn in self.synapses[nid]:
                    syn.apply_stdp(spiking_neuron, t)

                if enable_synaptogenesis:
                    self._attempt_synaptogenesis(nid, t)

    def train(self, patterns: Dict[str, List[np.ndarray]]):
        """
        Train the network on named spatiotemporal pattern sequences.

        Args:
            patterns: Dict mapping names to lists of temporal pattern steps
        """
        epochs = self.params.training_epochs

        print(f"Training on {len(patterns)} patterns for {epochs} epochs...")
        print(f"Network: {len(self.neurons)} neurons "
              f"({sum(1 for n in self.neurons if n.layer == 0)} input, "
              f"{sum(1 for n in self.neurons if n.layer > 0)} processing)")
        print(f"Initial synapses: {self.get_synapse_count()}")

        self.weight_snapshots.append(self.get_mean_excitatory_weight())
        self.atp_snapshots.append(self.get_mean_atp())
        self.synapse_count_snapshots.append(self.get_synapse_count())

        for epoch in range(epochs):
            epoch_spikes = 0

            for name, sequence in patterns.items():
                self.reset_neuron_states()
                spike_count_before = len(self.spike_history)
                self._run_pattern_sequence(sequence)
                epoch_spikes += len(self.spike_history) - spike_count_before

            mean_w = self.get_mean_excitatory_weight()
            mean_atp = self.get_mean_atp()
            syn_count = self.get_synapse_count()
            self.weight_snapshots.append(mean_w)
            self.atp_snapshots.append(mean_atp)
            self.synapse_count_snapshots.append(syn_count)

            print(f"  Epoch {epoch + 1:2d}/{epochs} | "
                  f"Spikes: {epoch_spikes:5d} | "
                  f"Weight: {mean_w:.4f} | "
                  f"ATP: {mean_atp:.3f} mM | "
                  f"Synapses: {syn_count}")

        print("Training complete.")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(network: LayeredNetwork):
    """Visualise training results with structural plasticity tracking."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Weight Evolution ---
    ax = axes[0, 0]
    ax.plot(network.weight_snapshots, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Excitatory Weight')
    ax.set_title('Weight Evolution')
    ax.axhline(y=network.weight_snapshots[0], color='r', linestyle='--',
               alpha=0.5, label='Initial')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Synapse Count Evolution ---
    ax = axes[0, 1]
    ax.plot(network.synapse_count_snapshots, 'm-s', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Synapses')
    ax.set_title('Synapse Count (Synaptogenesis)')
    ax.axhline(y=network.synapse_count_snapshots[0], color='r', linestyle='--',
               alpha=0.5, label='Initial')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Spike Raster (last epoch, coloured by layer) ---
    ax = axes[1, 0]
    if network.spike_history:
        spikes_per_epoch = len(network.spike_history) // max(network.params.training_epochs, 1)
        last_epoch_spikes = network.spike_history[-spikes_per_epoch:]
        if last_epoch_spikes:
            times, ids = zip(*last_epoch_spikes)
            layer_colors = {0: '#2196F3', 1: '#4CAF50', 2: '#FF9800'}
            colors = [layer_colors.get(network.neurons[nid].layer, 'black')
                      for nid in ids]
            ax.scatter(times, ids, s=2, c=colors, marker='|')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster (Last Epoch, by Layer)')
    ax.grid(True, alpha=0.3)

    # --- Layer Statistics ---
    ax = axes[1, 1]
    layer_ids = sorted(set(n.layer for n in network.neurons))
    layer_names = [f'Layer {l}' for l in layer_ids]
    layer_spike_counts = []
    layer_neuron_counts = []
    for l in layer_ids:
        neurons_in_layer = [n for n in network.neurons if n.layer == l]
        total_spikes = sum(len(n.spike_times) for n in neurons_in_layer)
        layer_spike_counts.append(total_spikes)
        layer_neuron_counts.append(len(neurons_in_layer))

    x = np.arange(len(layer_ids))
    width = 0.35
    ax.bar(x - width / 2, layer_neuron_counts, width, label='Neurons', color='#90CAF9')
    ax.bar(x + width / 2, [s // max(network.params.training_epochs, 1)
                            for s in layer_spike_counts],
           width, label='Spikes/epoch', color='#EF9A9A')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('Count')
    ax.set_title('Layer Composition and Activity')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('level4_structure_results.png', dpi=150)
    print("Results saved to: level4_structure_results.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=== Level 4: Structural Cortical Network ===\n")

    # Configure simulation
    sim = SimulationParams(
        dt=0.01,
        duration=100.0,           # Time per temporal step (ms)
        n_neurons=256,
        n_input_neurons=128,
        excitatory_ratio=0.8,
        input_current=25.0,
        training_epochs=5,
        synaptogenesis_rate=0.0005,
    )

    # Configure metabolism
    metabolism = MetabolicParams(
        blood_glucose=5.0,
        blood_ketones=0.1,
        complex_i_efficiency=1.0,
        atp_consumption_rate=1e-5,
    )

    # Create spatiotemporal training patterns
    patterns: Dict[str, List[np.ndarray]] = {}
    for label in ['A', 'B']:
        base = create_sparse_pattern(label, sim.n_input_neurons, n_active=16)
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(label))
        sequence = create_temporal_sequence(base, n_steps=4, n_flips=4, seed=seed)
        patterns[label] = sequence

    print("Patterns created:")
    for name, seq in patterns.items():
        active_counts = [int(np.sum(s)) for s in seq]
        print(f"  {name}: {len(seq)} temporal steps, "
              f"active neurons per step: {active_counts}")
    print()

    # Create and train network
    network = LayeredNetwork(sim, metabolism)

    # Layer summary
    for layer in sorted(set(n.layer for n in network.neurons)):
        neurons_in_layer = [n for n in network.neurons if n.layer == layer]
        n_pyr = sum(1 for n in neurons_in_layer if isinstance(n, PyramidalNeuron))
        n_bsk = sum(1 for n in neurons_in_layer if isinstance(n, BasketCell))
        print(f"  Layer {layer}: {len(neurons_in_layer)} neurons "
              f"({n_pyr} pyramidal, {n_bsk} basket)")

    initial_w = network.get_mean_excitatory_weight()
    initial_syn = network.get_synapse_count()
    print(f"\nInitial state: weight={initial_w:.4f}, synapses={initial_syn}\n")

    network.train(patterns)

    # Results
    final_w = network.get_mean_excitatory_weight()
    final_syn = network.get_synapse_count()

    print(f"\n=== Results ===")
    print(f"Initial synapses: {initial_syn}")
    print(f"Final   synapses: {final_syn} ({final_syn - initial_syn:+d} new)")
    print(f"Initial mean excitatory weight: {initial_w:.4f}")
    print(f"Final   mean excitatory weight: {final_w:.4f}")
    print(f"Change: {final_w - initial_w:+.4f}")

    # Per-layer spike summary
    print(f"\nPer-layer activity:")
    for layer in sorted(set(n.layer for n in network.neurons)):
        neurons_in_layer = [n for n in network.neurons if n.layer == layer]
        total_spikes = sum(len(n.spike_times) for n in neurons_in_layer)
        avg_spikes = total_spikes / len(neurons_in_layer) if neurons_in_layer else 0
        print(f"  Layer {layer}: {total_spikes} total spikes "
              f"({avg_spikes:.1f} per neuron)")

    # Visualize
    plot_results(network)
