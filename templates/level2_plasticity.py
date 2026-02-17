"""
Level 2: Adaptive Learning Networks
Builds on Level 1 by adding synaptic plasticity: STDP and Short-Term Plasticity.

The network can now LEARN from experience. Synaptic weights change based on the
precise timing of pre- and post-synaptic spikes (STDP), and short-term dynamics
(facilitation / depression) modulate signal transmission on fast timescales.

New concepts over Level 1:
  - Spike-Timing-Dependent Plasticity (STDP): long-term weight changes
  - Short-Term Plasticity (STP): facilitation and depression
  - Pattern-based training with multiple epochs
  - Weight evolution tracking and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import deque


# ============================================================================
# PARAMETERS
# ============================================================================

STDP_WINDOW = 20.0  # Time window for STDP (ms)

@dataclass
class NeuronParams:
    """Hodgkin-Huxley neuron parameters."""
    C_m: float = 1.0          # Membrane capacitance (uF/cm^2)
    V_rest: float = -65.0     # Resting potential (mV)
    g_Na: float = 120.0       # Sodium max conductance
    g_K: float = 36.0         # Potassium max conductance
    g_L: float = 0.3          # Leak conductance
    E_Na: float = 50.0        # Sodium reversal (mV)
    E_K: float = -77.0        # Potassium reversal (mV)
    E_L: float = -54.387      # Leak reversal (mV)


@dataclass
class SynapseParams:
    """Synapse parameters including plasticity settings."""
    weight: float = 3.0           # Initial synaptic strength
    max_weight: float = 7.5       # Upper bound on weight (prevents runaway)
    min_weight: float = 0.1       # Lower bound on weight
    delay: float = 1.0            # Propagation delay (ms)
    pulse_duration: float = 2.0   # Postsynaptic current duration (ms)
    is_inhibitory: bool = False   # True for GABAergic synapses

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


# ============================================================================
# NEURON
# ============================================================================

class HodgkinHuxleyNeuron:
    """Hodgkin-Huxley neuron model with spike history for plasticity."""

    def __init__(self, neuron_id: int, is_excitatory: bool, params: NeuronParams):
        self.id = neuron_id
        self.is_excitatory = is_excitatory
        self.params = params

        # State
        self.V = params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
        self.spike_times: List[float] = []

    def reset_state(self):
        """Reset dynamic state for a new simulation run, keeping learned weights."""
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
        """Advance neuron by one time step. Returns True if spiked."""
        p = self.params
        I_total = I_external + I_synaptic

        I_Na = p.g_Na * (self.m ** 3) * self.h * (self.V - p.E_Na)
        I_K = p.g_K * (self.n ** 4) * (self.V - p.E_K)
        I_L = p.g_L * (self.V - p.E_L)

        dV_dt = (I_total - I_Na - I_K - I_L) / p.C_m
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


# ============================================================================
# SYNAPSE WITH PLASTICITY (New in Level 2)
# ============================================================================

class PlasticSynapse:
    """
    Synapse with both short-term and long-term plasticity.

    Short-Term Plasticity (STP):
        Modulates effective transmission strength on fast timescales.
        - Facilitation: repeated use temporarily strengthens the synapse
        - Depression: repeated use temporarily weakens the synapse

    Spike-Timing-Dependent Plasticity (STDP):
        Adjusts the base weight based on relative spike timing.
        - LTP: pre fires before post -> weight increases
        - LTD: pre fires after post  -> weight decreases
    """

    def __init__(self, pre_neuron: HodgkinHuxleyNeuron, params: SynapseParams):
        self.pre_neuron = pre_neuron
        self.params = params
        self.weight = params.weight

        # Spike arrival queue
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
        # Time since last presynaptic spike
        dt = t - self.last_pre_spike_time if self.last_pre_spike_time > 0 else 0.0
        self.last_pre_spike_time = t

        p = self.params

        # Decay STP factors toward baseline (1.0) since last spike
        self.facilitation = 1.0 + (self.facilitation - 1.0) * np.exp(-dt / p.tau_facilitation)
        self.depression = 1.0 - (1.0 - self.depression) * np.exp(-dt / p.tau_depression)

        # Each spike pushes facilitation up and depression down
        self.facilitation += p.U_facilitation * (1.0 - self.facilitation)
        self.depression *= (1.0 - p.U_depression)

        # Schedule spike arrival after delay
        self.spike_queue.append(t + p.delay)

    def get_current(self, t: float) -> float:
        """Get postsynaptic current at time t, modulated by STP."""
        # Check for arriving spikes
        while self.spike_queue and self.spike_queue[0] <= t:
            self.spike_queue.popleft()
            self.pulse_end_time = t + self.params.pulse_duration

        if t < self.pulse_end_time:
            # Effective weight is base weight scaled by STP factors
            effective_weight = self.weight * self.facilitation * self.depression
            return -effective_weight if self.params.is_inhibitory else effective_weight

        return 0.0

    def apply_stdp(self, post_spike_time: float):
        """
        Apply STDP learning rule on a postsynaptic spike.

        Called by the postsynaptic neuron when it fires. Adjusts the base
        weight based on the time difference between pre and post spikes.
        """
        # Only excitatory synapses undergo STDP in this model
        if self.params.is_inhibitory or self.last_pre_spike_time < 0:
            return

        self.last_post_spike_time = post_spike_time
        delta_t = post_spike_time - self.last_pre_spike_time
        delta_w = 0.0
        p = self.params

        if 0 < delta_t < STDP_WINDOW:
            # LTP: pre before post -> strengthen
            delta_w = p.stdp_ltp_rate * np.exp(-delta_t / STDP_WINDOW)
        elif -STDP_WINDOW < delta_t < 0:
            # LTD: pre after post -> weaken
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
# PATTERN UTILITIES (New in Level 2)
# ============================================================================

def create_sparse_pattern(label: str, size: int, n_active: int = 8) -> np.ndarray:
    """
    Create a sparse binary pattern for a given label.

    Uses the label as a seed so the same label always produces the same
    pattern, enabling reproducible experiments.

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


# ============================================================================
# NETWORK WITH LEARNING
# ============================================================================

class PlasticNetwork:
    """
    Network of neurons connected by plastic synapses.

    Extends the Level 1 NeuralNetwork with:
      - Excitatory / inhibitory neuron types
      - PlasticSynapse connections (STDP + STP)
      - Pattern-based training interface
      - Weight tracking for analysis
    """

    def __init__(self, sim_params: SimulationParams):
        self.params = sim_params
        neuron_params = NeuronParams()

        n_excitatory = int(sim_params.n_neurons * sim_params.excitatory_ratio)

        # Create neurons with excitatory / inhibitory identity
        self.neurons: List[HodgkinHuxleyNeuron] = []
        for i in range(sim_params.n_neurons):
            is_exc = i < n_excitatory
            self.neurons.append(HodgkinHuxleyNeuron(i, is_exc, neuron_params))

        # Build synaptic connections
        # synapses[post_id] = list of PlasticSynapse incoming to that neuron
        self.synapses: Dict[int, List[PlasticSynapse]] = {
            i: [] for i in range(sim_params.n_neurons)
        }
        self._create_connections()

        # Recording
        self.spike_history: List[tuple] = []
        self.weight_snapshots: List[float] = []

    def _create_connections(self):
        """Create random connectivity with appropriate synapse types."""
        p = self.params
        exc_params = SynapseParams(
            weight=3.0,
            max_weight=7.5,
            is_inhibitory=False,
        )
        inh_params = SynapseParams(
            weight=5.0,
            max_weight=12.5,
            is_inhibitory=True,
        )

        for post_neuron in self.neurons:
            for pre_neuron in self.neurons:
                if pre_neuron.id == post_neuron.id:
                    continue
                if np.random.random() < p.connection_prob:
                    params = exc_params if pre_neuron.is_excitatory else inh_params
                    syn = PlasticSynapse(pre_neuron, params)
                    self.synapses[post_neuron.id].append(syn)

    def get_mean_excitatory_weight(self) -> float:
        """Calculate the mean weight of all excitatory synapses."""
        weights = []
        for syn_list in self.synapses.values():
            for syn in syn_list:
                if not syn.params.is_inhibitory:
                    weights.append(syn.weight)
        return float(np.mean(weights)) if weights else 0.0

    def reset_neuron_states(self):
        """Reset all neuron and synapse dynamic state between patterns."""
        for neuron in self.neurons:
            neuron.reset_state()
        for syn_list in self.synapses.values():
            for syn in syn_list:
                syn.reset_dynamic_state()

    def _run_single_pattern(self, pattern: np.ndarray):
        """Present one pattern and run the simulation for one duration."""
        dt = self.params.dt
        n_steps = int(self.params.duration / dt)

        for step in range(n_steps):
            t = step * dt
            spiked_neurons: List[int] = []

            for neuron in self.neurons:
                # External input: apply current to input neurons where pattern bit is 1
                I_ext = 0.0
                if neuron.id < self.params.n_input_neurons:
                    if neuron.id < len(pattern) and pattern[neuron.id] == 1:
                        I_ext = self.params.input_current

                # Synaptic input from all incoming connections
                I_syn = sum(syn.get_current(t) for syn in self.synapses[neuron.id])

                spiked = neuron.step(dt, I_ext, I_syn, t)

                if spiked:
                    spiked_neurons.append(neuron.id)
                    self.spike_history.append((t, neuron.id))

            # Propagate spikes and apply plasticity
            for nid in spiked_neurons:
                # Notify all downstream synapses of the presynaptic spike
                for post_id, syn_list in self.synapses.items():
                    for syn in syn_list:
                        if syn.pre_neuron.id == nid:
                            syn.on_presynaptic_spike(t)

                # Apply STDP on all incoming synapses to the spiking neuron
                for syn in self.synapses[nid]:
                    syn.apply_stdp(t)

    def train(self, patterns: Dict[str, np.ndarray]):
        """
        Train the network on a set of named patterns over multiple epochs.

        Args:
            patterns: Dictionary mapping pattern names to binary arrays
        """
        epochs = self.params.training_epochs

        print(f"Training on {len(patterns)} patterns for {epochs} epochs...")
        print(f"Network: {self.params.n_neurons} neurons, "
              f"{sum(len(s) for s in self.synapses.values())} synapses")

        # Record initial weight state
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
# VISUALIZATION
# ============================================================================

def plot_results(network: PlasticNetwork, patterns: Dict[str, np.ndarray]):
    """Visualise training results: weight evolution, spike raster, pattern overlap."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Weight Evolution ---
    ax = axes[0, 0]
    ax.plot(network.weight_snapshots, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Excitatory Weight')
    ax.set_title('Weight Evolution During Training')
    ax.axhline(y=network.weight_snapshots[0], color='r', linestyle='--',
               alpha=0.5, label='Initial weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Weight Distribution ---
    ax = axes[0, 1]
    exc_weights = []
    for syn_list in network.synapses.values():
        for syn in syn_list:
            if not syn.params.is_inhibitory:
                exc_weights.append(syn.weight)
    ax.hist(exc_weights, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Synaptic Weight')
    ax.set_ylabel('Count')
    ax.set_title('Final Excitatory Weight Distribution')
    ax.grid(True, alpha=0.3)

    # --- Spike Raster (last epoch) ---
    ax = axes[1, 0]
    if network.spike_history:
        # Show only last epoch's spikes
        n_patterns = len(patterns)
        spikes_per_epoch = len(network.spike_history) // max(network.params.training_epochs, 1)
        last_epoch_spikes = network.spike_history[-spikes_per_epoch:]
        if last_epoch_spikes:
            times, ids = zip(*last_epoch_spikes)
            colors = ['black' if network.neurons[nid].is_excitatory else 'red'
                      for nid in ids]
            ax.scatter(times, ids, s=3, c=colors, marker='|')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster (Last Epoch)')
    ax.grid(True, alpha=0.3)

    # --- Pattern Overlap Analysis ---
    ax = axes[1, 1]
    pattern_names = list(patterns.keys())
    n_pat = len(pattern_names)
    overlap_matrix = np.zeros((n_pat, n_pat))
    for i, name_i in enumerate(pattern_names):
        for j, name_j in enumerate(pattern_names):
            p_i = patterns[name_i]
            p_j = patterns[name_j]
            min_len = min(len(p_i), len(p_j))
            overlap_matrix[i, j] = np.sum(p_i[:min_len] & p_j[:min_len])
    im = ax.imshow(overlap_matrix, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(n_pat))
    ax.set_yticks(range(n_pat))
    ax.set_xticklabels(pattern_names)
    ax.set_yticklabels(pattern_names)
    ax.set_title('Pattern Overlap (shared active bits)')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('level2_plasticity_results.png', dpi=150)
    print("Results saved to: level2_plasticity_results.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configure
    sim = SimulationParams(
        dt=0.01,
        duration=250.0,
        n_neurons=50,
        excitatory_ratio=0.8,
        connection_prob=0.1,
        input_current=25.0,
        n_input_neurons=32,
        training_epochs=10,
    )

    # Create network
    network = PlasticNetwork(sim)

    # Create training patterns
    patterns = {
        'A': create_sparse_pattern('A', sim.n_input_neurons, n_active=8),
        'B': create_sparse_pattern('B', sim.n_input_neurons, n_active=8),
        'C': create_sparse_pattern('C', sim.n_input_neurons, n_active=8),
    }

    print("=== Level 2: Adaptive Learning Network ===\n")
    print("Patterns created:")
    for name, pat in patterns.items():
        active = np.where(pat == 1)[0]
        print(f"  {name}: active neurons {active.tolist()}")
    print()

    initial_weight = network.get_mean_excitatory_weight()
    print(f"Initial mean excitatory weight: {initial_weight:.4f}\n")

    # Train
    network.train(patterns)

    # Analyse results
    final_weight = network.get_mean_excitatory_weight()
    print(f"\n=== Results ===")
    print(f"Initial mean excitatory weight: {initial_weight:.4f}")
    print(f"Final   mean excitatory weight: {final_weight:.4f}")
    print(f"Change: {final_weight - initial_weight:+.4f}")

    if final_weight > initial_weight:
        print("SUCCESS: Synaptic weights potentiated - learning occurred.")
    else:
        print("NOTE: Weights did not show net potentiation.")

    # Per-neuron spike summary
    print(f"\nSpike summary (top 10 most active):")
    spike_counts = [(n.id, len(n.spike_times), 'E' if n.is_excitatory else 'I')
                    for n in network.neurons if n.spike_times]
    spike_counts.sort(key=lambda x: x[1], reverse=True)
    for nid, count, ntype in spike_counts[:10]:
        print(f"  Neuron {nid:3d} ({ntype}): {count} spikes")

    # Visualise
    plot_results(network, patterns)
