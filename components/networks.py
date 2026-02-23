"""
Network models: from basic circuits to layered cortical architectures.

Each network class composes neurons and synapses at the matching complexity level.
  - NeuralNetwork: Static HH circuit (Level 1)
  - PlasticNetwork: STDP + STP learning (Level 2)
  - MetabolicNetwork: ATP-coupled dynamics with AGC (Level 3)
  - LayeredNetwork: Layered architecture with synaptogenesis (Level 4)

Level 3+ networks include ATP-gated Automatic Gain Control (AGC):
  - Homeostatic firing-rate regulation (Turrigiano & Nelson 2004)
  - ATP gating prevents AGC from masking metabolic failure
  - metabolic_gate(ATP) = 1/(1+exp(-(ATP-half)/slope))
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .base import (
    NeuronParams, MetabolicParams, SynapseParams, SimulationParams,
    AGCParams, Vector3, STDP_WINDOW, create_sparse_pattern,
)
from .neurons import (
    HodgkinHuxleyNeuron, MetabolicNeuron,
    StructuralNeuron, PyramidalNeuron, BasketCell,
)
from .synapses import (
    Synapse, PlasticSynapse, MetabolicPlasticSynapse, StructuralSynapse,
)


# ============================================================================
# AGC HELPER
# ============================================================================

def metabolic_gate_agc(mean_atp: float, agc_params: AGCParams) -> float:
    """
    Sigmoidal gate: AGC effectiveness as a function of mean network ATP.

    At normal ATP (~3 mM): gate ~ 0.95 (full homeostatic control)
    At low ATP (~1.5 mM): gate ~ 0.50 (weakened control)
    At depleted ATP (~0.5 mM): gate ~ 0.12 (near-failure)
    """
    return 1.0 / (1.0 + np.exp(
        -(mean_atp - agc_params.atp_half) / agc_params.atp_slope
    ))


# ============================================================================
# LEVEL 1: BASIC NETWORK
# ============================================================================

class NeuralNetwork:
    """
    Network of Hodgkin-Huxley neurons with synaptic connections.

    Creates random connectivity and runs a continuous simulation with
    external current injection.
    """

    def __init__(self, sim_params: SimulationParams):
        self.params = sim_params
        neuron_params = NeuronParams()

        self.neurons = [
            HodgkinHuxleyNeuron(i, True, neuron_params)
            for i in range(sim_params.n_neurons)
        ]

        self.synapses: Dict[int, List[Synapse]] = {
            i: [] for i in range(sim_params.n_neurons)
        }
        self._create_random_connections()

        self.voltage_history: List[Tuple[float, list]] = []
        self.spike_history: List[Tuple[float, int]] = []

    def _create_random_connections(self):
        synapse_params = SynapseParams()
        for post_neuron in self.neurons:
            for pre_neuron in self.neurons:
                if pre_neuron.id == post_neuron.id:
                    continue
                if np.random.random() < self.params.connection_prob:
                    syn = Synapse(pre_neuron, synapse_params)
                    self.synapses[post_neuron.id].append(syn)

    def run(self, external_input_neurons: Optional[List[int]] = None):
        """Run the simulation with external current on specified neurons."""
        if external_input_neurons is None:
            external_input_neurons = [0, 1]

        dt = self.params.dt
        n_steps = int(self.params.duration / dt)

        print(f"Running simulation for {self.params.duration} ms...")
        print(f"Network: {self.params.n_neurons} neurons, "
              f"{sum(len(s) for s in self.synapses.values())} synapses")

        for step in range(n_steps):
            t = step * dt
            voltages = []

            for neuron in self.neurons:
                I_ext = self.params.input_current if neuron.id in external_input_neurons else 0.0
                I_syn = sum(syn.get_current(t) for syn in self.synapses.get(neuron.id, []))

                spiked = neuron.step(dt, I_ext, I_syn, t)
                if spiked:
                    self.spike_history.append((t, neuron.id))
                    for post_id in range(len(self.neurons)):
                        for syn in self.synapses[post_id]:
                            if syn.pre_neuron.id == neuron.id:
                                syn.on_presynaptic_spike(t)

                voltages.append(neuron.V)

            if step % 10 == 0:
                self.voltage_history.append((t, voltages.copy()))

        print(f"Simulation complete! Total spikes: {len(self.spike_history)}")


# ============================================================================
# LEVEL 2: PLASTIC NETWORK
# ============================================================================

class PlasticNetwork:
    """
    Network of neurons connected by plastic synapses (STDP + STP).

    Supports pattern-based training over multiple epochs with weight tracking.
    """

    def __init__(self, sim_params: SimulationParams):
        self.params = sim_params
        neuron_params = NeuronParams()
        n_excitatory = int(sim_params.n_neurons * sim_params.excitatory_ratio)

        self.neurons: List[HodgkinHuxleyNeuron] = []
        for i in range(sim_params.n_neurons):
            is_exc = i < n_excitatory
            self.neurons.append(HodgkinHuxleyNeuron(i, is_exc, neuron_params))

        self.synapses: Dict[int, List[PlasticSynapse]] = {
            i: [] for i in range(sim_params.n_neurons)
        }
        self._create_connections()

        self.spike_history: List[Tuple[float, int]] = []
        self.weight_snapshots: List[float] = []

    def _create_connections(self):
        # E:I weight ratio ~ 1:4 (Markram et al. 2015)
        exc_params = SynapseParams(weight=1.0, max_weight=2.5, is_inhibitory=False)
        inh_params = SynapseParams(weight=4.0, max_weight=10.0, is_inhibitory=True)

        for post_neuron in self.neurons:
            for pre_neuron in self.neurons:
                if pre_neuron.id == post_neuron.id:
                    continue
                if np.random.random() < self.params.connection_prob:
                    params = exc_params if pre_neuron.is_excitatory else inh_params
                    syn = PlasticSynapse(pre_neuron, params)
                    self.synapses[post_neuron.id].append(syn)

    def get_mean_excitatory_weight(self) -> float:
        weights = [syn.weight for sl in self.synapses.values()
                   for syn in sl if not syn.params.is_inhibitory]
        return float(np.mean(weights)) if weights else 0.0

    def reset_neuron_states(self):
        for neuron in self.neurons:
            neuron.reset_state()
        for syn_list in self.synapses.values():
            for syn in syn_list:
                syn.reset_dynamic_state()

    def _run_single_pattern(self, pattern: np.ndarray):
        dt = self.params.dt
        n_steps = int(self.params.duration / dt)

        for step in range(n_steps):
            t = step * dt
            spiked_neurons: List[int] = []

            for neuron in self.neurons:
                I_ext = 0.0
                if neuron.id < self.params.n_input_neurons:
                    if neuron.id < len(pattern) and pattern[neuron.id] == 1:
                        I_ext = self.params.input_current

                I_syn = sum(syn.get_current(t) for syn in self.synapses[neuron.id])
                spiked = neuron.step(dt, I_ext, I_syn, t)

                if spiked:
                    spiked_neurons.append(neuron.id)
                    self.spike_history.append((t, neuron.id))

            for nid in spiked_neurons:
                for post_id, syn_list in self.synapses.items():
                    for syn in syn_list:
                        if syn.pre_neuron.id == nid:
                            syn.on_presynaptic_spike(t)
                for syn in self.synapses[nid]:
                    syn.apply_stdp(t)

    def train(self, patterns: Dict[str, np.ndarray]):
        """Train the network on named patterns over multiple epochs."""
        epochs = self.params.training_epochs

        print(f"Training on {len(patterns)} patterns for {epochs} epochs...")
        print(f"Network: {self.params.n_neurons} neurons, "
              f"{sum(len(s) for s in self.synapses.values())} synapses")

        self.weight_snapshots.append(self.get_mean_excitatory_weight())

        for epoch in range(epochs):
            epoch_spikes = 0
            for name, pattern in patterns.items():
                self.reset_neuron_states()
                spike_count_before = len(self.spike_history)
                self._run_single_pattern(pattern)
                epoch_spikes += len(self.spike_history) - spike_count_before

            mean_w = self.get_mean_excitatory_weight()
            self.weight_snapshots.append(mean_w)
            print(f"  Epoch {epoch + 1:2d}/{epochs} | "
                  f"Spikes: {epoch_spikes:5d} | "
                  f"Mean excitatory weight: {mean_w:.4f}")

        print("Training complete.")


# ============================================================================
# LEVEL 3: METABOLIC NETWORK (with AGC)
# ============================================================================

class MetabolicNetwork:
    """
    Network of metabolic neurons with energy-dependent plastic synapses
    and ATP-gated automatic gain control (AGC).

    AGC maintains a target firing rate by adjusting input gain, but its
    effectiveness is modulated by metabolic state. Under low ATP, AGC
    weakens, allowing metabolic failure to impact network activity.
    """

    def __init__(self, sim_params: SimulationParams,
                 metabolic_params: MetabolicParams = None,
                 agc_params: AGCParams = None):
        self.params = sim_params
        self.metabolic_params = metabolic_params or MetabolicParams()
        self.agc_params = agc_params or AGCParams()
        neuron_params = NeuronParams()
        n_excitatory = int(sim_params.n_neurons * sim_params.excitatory_ratio)

        self.neurons: List[MetabolicNeuron] = []
        for i in range(sim_params.n_neurons):
            is_exc = i < n_excitatory
            self.neurons.append(
                MetabolicNeuron(i, is_exc, neuron_params, self.metabolic_params)
            )

        self.synapses: Dict[int, List[MetabolicPlasticSynapse]] = {
            i: [] for i in range(sim_params.n_neurons)
        }
        self._create_connections()

        # AGC state
        self.current_gain = self.agc_params.initial_gain
        self.recent_spike_count = 0
        self.last_agc_time = 0.0

        # History tracking
        self.spike_history: List[Tuple[float, int]] = []
        self.weight_snapshots: List[float] = []
        self.atp_snapshots: List[float] = []
        self.gain_history: List[float] = []

    def _create_connections(self):
        # E:I weight ratio ~ 1:4
        exc_params = SynapseParams(weight=1.0, max_weight=2.5, is_inhibitory=False)
        inh_params = SynapseParams(weight=4.0, max_weight=10.0, is_inhibitory=True)

        for post_neuron in self.neurons:
            for pre_neuron in self.neurons:
                if pre_neuron.id == post_neuron.id:
                    continue
                if np.random.random() < self.params.connection_prob:
                    params = exc_params if pre_neuron.is_excitatory else inh_params
                    syn = MetabolicPlasticSynapse(pre_neuron, params)
                    self.synapses[post_neuron.id].append(syn)

    def get_mean_excitatory_weight(self) -> float:
        weights = [syn.weight for sl in self.synapses.values()
                   for syn in sl if not syn.params.is_inhibitory]
        return float(np.mean(weights)) if weights else 0.0

    def get_mean_atp(self) -> float:
        return float(np.mean([n.metabolism.get_atp() for n in self.neurons]))

    def get_synapse_count(self) -> int:
        return sum(len(s) for s in self.synapses.values())

    def _update_agc(self, t: float):
        """Update AGC gain based on recent firing rate and metabolic state."""
        agc = self.agc_params
        if not agc.enabled:
            return

        if t - self.last_agc_time >= agc.interval_ms:
            n_neurons = len(self.neurons)
            observed_rate = (
                (self.recent_spike_count / n_neurons) *
                (1000.0 / agc.interval_ms)
            )

            error = agc.target_rate_hz - observed_rate
            gate = metabolic_gate_agc(self.get_mean_atp(), agc)
            effective_lambda = agc.lambda_base * gate

            self.current_gain += effective_lambda * error
            self.current_gain = np.clip(self.current_gain, agc.gain_min, agc.gain_max)

            self.gain_history.append(self.current_gain)
            self.recent_spike_count = 0
            self.last_agc_time = t

    def reset_neuron_states(self):
        for neuron in self.neurons:
            neuron.reset_state()
        for syn_list in self.synapses.values():
            for syn in syn_list:
                syn.reset_dynamic_state()

    def _run_single_pattern(self, pattern: np.ndarray):
        dt = self.params.dt
        n_steps = int(self.params.duration / dt)

        for step in range(n_steps):
            t = step * dt
            spiked_neurons: List[int] = []

            self._update_agc(t)

            for neuron in self.neurons:
                I_ext = 0.0
                if neuron.id < self.params.n_input_neurons:
                    if neuron.id < len(pattern) and pattern[neuron.id] == 1:
                        I_ext = self.params.input_current

                I_syn = sum(syn.get_current(t) for syn in self.synapses[neuron.id])
                spiked = neuron.step(dt, I_ext, I_syn, t)

                if spiked:
                    spiked_neurons.append(neuron.id)
                    self.spike_history.append((t, neuron.id))
                    self.recent_spike_count += 1

            for nid in spiked_neurons:
                spiking_neuron = self.neurons[nid]
                for post_id, syn_list in self.synapses.items():
                    for syn in syn_list:
                        if syn.pre_neuron.id == nid:
                            syn.on_presynaptic_spike(t)
                for syn in self.synapses[nid]:
                    syn.apply_stdp(spiking_neuron, t)

    def train(self, patterns: Dict[str, np.ndarray]):
        """Train the network on named patterns over multiple epochs."""
        epochs = self.params.training_epochs

        print(f"Training on {len(patterns)} patterns for {epochs} epochs...")
        print(f"Network: {self.params.n_neurons} neurons, "
              f"{self.get_synapse_count()} synapses")
        print(f"Metabolic state: glucose={self.metabolic_params.blood_glucose:.1f} mM, "
              f"ketones={self.metabolic_params.blood_ketones:.1f} mM, "
              f"Complex I={self.metabolic_params.complex_i_efficiency:.0%}")
        print(f"AGC: {'enabled' if self.agc_params.enabled else 'disabled'}, "
              f"target={self.agc_params.target_rate_hz:.0f} Hz")

        self.weight_snapshots.append(self.get_mean_excitatory_weight())
        self.atp_snapshots.append(self.get_mean_atp())

        for epoch in range(epochs):
            epoch_spikes = 0
            for name, pattern in patterns.items():
                self.reset_neuron_states()
                spike_count_before = len(self.spike_history)
                self._run_single_pattern(pattern)
                epoch_spikes += len(self.spike_history) - spike_count_before

            mean_w = self.get_mean_excitatory_weight()
            mean_atp = self.get_mean_atp()
            self.weight_snapshots.append(mean_w)
            self.atp_snapshots.append(mean_atp)

            print(f"  Epoch {epoch + 1:2d}/{epochs} | "
                  f"Spikes: {epoch_spikes:5d} | "
                  f"Weight: {mean_w:.4f} | "
                  f"ATP: {mean_atp:.3f} mM | "
                  f"Gain: {self.current_gain:.2f}")

        print("Training complete.")


# ============================================================================
# LEVEL 4: LAYERED NETWORK (with AGC + ATP-gated synaptogenesis)
# ============================================================================

class LayeredNetwork:
    """
    Cortical-like network with layered architecture, cell type diversity,
    spatial organisation, structural plasticity, and homeostatic gain control.

    Architecture:
      - Layer 0 (Input): All pyramidal neurons, grid-positioned
      - Layer 1 (Processing): Mixed pyramidal + basket cells
      - Layer 2 (Integration): Mixed pyramidal + basket cells

    Features:
      - ATP-gated AGC (homeostatic firing-rate regulation)
      - ATP-gated synaptogenesis (GMR budget; Rae et al. 2024)
      - Distance-dependent connectivity (Markram et al. 2015)

    Connection probabilities from cortical microcircuit data:
      - Feedforward (L0->L1): 0.5 * exp(-d/250)
      - Local same-layer (<150 um): 0.3 * exp(-d/400)
      - Distant same-layer: 0.05 * exp(-d/400)
    """

    def __init__(self, sim_params: SimulationParams,
                 metabolic_params: MetabolicParams = None,
                 agc_params: AGCParams = None):
        self.params = sim_params
        self.metabolic_params = metabolic_params or MetabolicParams()
        self.agc_params = agc_params or AGCParams()
        neuron_params = NeuronParams()

        self.neurons: List[StructuralNeuron] = []
        self.synapses: Dict[int, List[StructuralSynapse]] = {}

        # AGC state
        self.current_gain = self.agc_params.initial_gain
        self.recent_spike_count = 0
        self.last_agc_time = 0.0

        # History tracking
        self.spike_history: List[Tuple[float, int]] = []
        self.weight_snapshots: List[float] = []
        self.atp_snapshots: List[float] = []
        self.synapse_count_snapshots: List[int] = []
        self.gain_history: List[float] = []
        self.last_spike_times: Dict[int, float] = {}

        self._create_neurons(neuron_params)
        self._create_layered_connections()

    def _create_neurons(self, neuron_params: NeuronParams):
        mp = self.metabolic_params
        n_input = self.params.n_input_neurons
        n_processing = (self.params.n_neurons - n_input) // 2
        n_integration = self.params.n_neurons - n_input - n_processing
        neuron_id = 0

        # Layer 0: Input - all pyramidal, grid positions
        for i in range(n_input):
            col = i % 16
            row = i // 16
            pos = Vector3((col - 8) * 20.0, (row - 4) * 20.0, 0.0)
            self.neurons.append(
                PyramidalNeuron(neuron_id, 0, pos, neuron_params, mp)
            )
            neuron_id += 1

        # Layer 1 and 2: mixed cell types
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

        for n in self.neurons:
            self.synapses[n.id] = []

    def _create_layered_connections(self):
        """
        Distance-dependent connectivity from cortical microcircuit data
        (Markram et al. 2015; Lefort et al. 2009).
        """
        FEEDFORWARD_PROB = 0.5
        LOCAL_PROB = 0.3
        DISTANT_PROB = 0.05

        # E:I weight ratio ~ 1:4
        exc_params = SynapseParams(weight=1.0, max_weight=2.5, is_inhibitory=False)
        inh_params = SynapseParams(weight=4.0, max_weight=10.0, is_inhibitory=True)

        for pre in self.neurons:
            for post in self.neurons:
                if pre.id == post.id:
                    continue

                distance = pre.position.distance_to(post.position)
                connection_prob = 0.0

                if pre.layer == 0 and post.layer == 1:
                    connection_prob = FEEDFORWARD_PROB * np.exp(-distance / 250.0)
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
        for syn in self.synapses[post_id]:
            if syn.pre_neuron.id == pre_id:
                return True
        return False

    def _attempt_synaptogenesis(self, post_idx: int, t: float):
        """
        Attempt to form new synapses via Hebbian synaptogenesis.

        ATP gating: probability scaled by mean ATP of pre/post neurons.
        Models the GMR energy budget from Rae et al. (2024).
        """
        post_neuron = self.neurons[post_idx]
        if post_neuron.layer == 0:
            return

        exc_params = SynapseParams(weight=0.2, max_weight=2.5, is_inhibitory=False)
        inh_params = SynapseParams(weight=0.3, max_weight=10.0, is_inhibitory=True)

        for pre_idx, pre_neuron in enumerate(self.neurons):
            if pre_idx == post_idx:
                continue

            pre_last_spike = self.last_spike_times.get(pre_idx, -np.inf)
            delta_t = t - pre_last_spike

            if 0 < delta_t < STDP_WINDOW:
                # ATP-gated synaptogenesis (GMR energy budget)
                mean_atp = (pre_neuron.metabolism.get_atp() +
                            post_neuron.metabolism.get_atp()) / 2.0
                atp_scaling = 1.0 / (1.0 + np.exp(-(mean_atp - 1.5) / 0.5))
                effective_rate = self.params.synaptogenesis_rate * atp_scaling

                if np.random.random() < effective_rate:
                    if not self._check_connection_exists(pre_idx, post_idx):
                        params = exc_params if pre_neuron.is_excitatory else inh_params
                        delay = pre_neuron.get_delay_to(post_neuron)
                        new_syn = StructuralSynapse(pre_neuron, params, delay)
                        self.synapses[post_idx].append(new_syn)

    def _update_agc(self, t: float):
        """Update AGC gain based on recent firing rate and metabolic state."""
        agc = self.agc_params
        if not agc.enabled:
            return

        if t - self.last_agc_time >= agc.interval_ms:
            n_neurons = len(self.neurons)
            observed_rate = (
                (self.recent_spike_count / n_neurons) *
                (1000.0 / agc.interval_ms)
            )

            error = agc.target_rate_hz - observed_rate
            gate = metabolic_gate_agc(self.get_mean_atp(), agc)
            effective_lambda = agc.lambda_base * gate

            self.current_gain += effective_lambda * error
            self.current_gain = np.clip(self.current_gain, agc.gain_min, agc.gain_max)

            self.gain_history.append(self.current_gain)
            self.recent_spike_count = 0
            self.last_agc_time = t

    def get_mean_excitatory_weight(self) -> float:
        weights = [syn.weight for sl in self.synapses.values()
                   for syn in sl if not syn.params.is_inhibitory]
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
        """Run a temporal pattern sequence through the network."""
        dt = self.params.dt
        step_duration = self.params.duration
        total_time = step_duration * len(sequence)
        n_total_steps = int(total_time / dt)

        for sim_step in range(n_total_steps):
            t = sim_step * dt
            pattern_idx = min(int(t / step_duration), len(sequence) - 1)
            current_pattern = sequence[pattern_idx]
            spiked_neurons: List[int] = []

            self._update_agc(t)

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
                    self.recent_spike_count += 1

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
        """Train the network on named spatiotemporal pattern sequences."""
        epochs = self.params.training_epochs

        print(f"Training on {len(patterns)} patterns for {epochs} epochs...")
        print(f"Network: {len(self.neurons)} neurons "
              f"({sum(1 for n in self.neurons if n.layer == 0)} input, "
              f"{sum(1 for n in self.neurons if n.layer > 0)} processing)")
        print(f"Initial synapses: {self.get_synapse_count()}")
        print(f"AGC: {'enabled' if self.agc_params.enabled else 'disabled'}, "
              f"target={self.agc_params.target_rate_hz:.0f} Hz")

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
                  f"Synapses: {syn_count} | "
                  f"Gain: {self.current_gain:.2f}")

        print("Training complete.")
