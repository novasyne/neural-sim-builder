"""
Level 3: Bioenergetic Networks
Builds on Level 2 by coupling neural function to cellular energy metabolism.

A MetabolicComponent is added to each neuron, simulating ATP production from
substrates like glucose and ketones. Neural activity is now energy-dependent:
ion pump efficiency, synaptic transmission, and STDP learning all scale with
the neuron's ATP supply. This allows the model to simulate how metabolic
state directly affects learning and computation.

New concepts over Level 2:
  - MetabolicComponent: ATP production via glycolysis and oxidative phosphorylation
  - Glucose and ketone metabolism as fuel sources
  - Energy-coupled Hodgkin-Huxley dynamics (ATP-dependent ion pumps)
  - Metabolism-modulated STDP (learning scales with ATP)
  - Metabolism-modulated synaptic transmission
  - ATP level tracking and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
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
    tau_facilitation: float = 200.0
    tau_depression: float = 500.0
    U_facilitation: float = 0.1
    U_depression: float = 0.2


@dataclass
class SimulationParams:
    """Global simulation parameters."""
    dt: float = 0.01              # Time step (ms)
    duration: float = 250.0       # Simulation time per pattern (ms)
    n_neurons: int = 50           # Number of neurons
    excitatory_ratio: float = 0.8
    connection_prob: float = 0.1
    input_current: float = 25.0
    n_input_neurons: int = 32
    training_epochs: int = 10


# ============================================================================
# METABOLIC COMPONENT (New in Level 3)
# ============================================================================

class MetabolicComponent:
    """
    Models core neuroenergetic pathways within a neuron.

    ATP is produced from two substrate pathways:
      - Glycolysis: glucose -> 2 ATP (fast, anaerobic)
      - Oxidative phosphorylation (OXPHOS): glucose/ketones -> 28 ATP
        (slow, requires mitochondria with functional Complex I)

    ATP is consumed by ion pumps (Na+/K+-ATPase) that restore membrane
    gradients after spiking activity.

    When ATP drops, the neuron's ability to maintain ion gradients, transmit
    signals, and perform synaptic plasticity is directly impaired.
    """

    # Metabolic rate constants
    K_GLYCOLYSIS_ATP = 2.0     # ATP yield from glycolysis per glucose
    K_OXPHOS_ATP = 28.0        # ATP yield from oxidative phosphorylation
    K_SUBSTRATE_UPTAKE = 0.01  # Substrate uptake rate constant

    def __init__(self, params: MetabolicParams):
        self.atp = params.initial_atp
        self.complex_i_efficiency = params.complex_i_efficiency

    def update(self, dt: float, atp_consumption: float,
               blood_glucose: float, blood_ketones: float):
        """
        Update ATP based on production and consumption.

        Args:
            dt: Time step (ms)
            atp_consumption: ATP consumed by ion pumps this step
            blood_glucose: Systemic glucose concentration (mM)
            blood_ketones: Systemic ketone concentration (mM)
        """
        glucose_flux = self.K_SUBSTRATE_UPTAKE * blood_glucose
        ketone_flux = self.K_SUBSTRATE_UPTAKE * blood_ketones

        atp_generated = (
            glucose_flux * self.K_GLYCOLYSIS_ATP +
            (glucose_flux + ketone_flux) * self.K_OXPHOS_ATP * self.complex_i_efficiency
        )

        self.atp += atp_generated * dt - atp_consumption
        self.atp = max(0.01, self.atp)  # Never fully depleted

    def get_atp(self) -> float:
        return self.atp


# ============================================================================
# NEURON WITH METABOLISM (Extended in Level 3)
# ============================================================================

class MetabolicNeuron:
    """
    Hodgkin-Huxley neuron with integrated bioenergetics.

    Extends the Level 2 neuron by coupling ion pump efficiency to ATP levels.
    When ATP is low, the effective potassium reversal potential shifts toward
    V_rest, reducing the neuron's ability to repolarize and fire cleanly.
    """

    def __init__(self, neuron_id: int, is_excitatory: bool,
                 neuron_params: NeuronParams, metabolic_params: MetabolicParams):
        self.id = neuron_id
        self.is_excitatory = is_excitatory
        self.params = neuron_params
        self.metabolic_params = metabolic_params
        self.metabolism = MetabolicComponent(metabolic_params)

        # Membrane state
        self.V = neuron_params.V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.is_spiking = False
        self.spike_times: List[float] = []

    def reset_state(self):
        """Reset dynamic state for a new simulation run."""
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
        """
        Advance neuron by one time step with energy-coupled dynamics.

        The key metabolic coupling is in the potassium current: when ATP is low,
        the Na+/K+-ATPase cannot fully restore ion gradients, so E_K shifts
        toward V_rest. This degrades spike quality and firing regularity.

        Returns True if the neuron spiked this step.
        """
        p = self.params
        mp = self.metabolic_params
        I_total = I_external + I_synaptic

        # Ionic currents
        I_Na = p.g_Na * (self.m ** 3) * self.h * (self.V - p.E_Na)
        I_K_ideal = p.g_K * (self.n ** 4) * (self.V - p.E_K)
        I_L = p.g_L * (self.V - p.E_L)

        # ATP consumption: proportional to ionic current magnitudes
        atp_consumed = (abs(I_Na) + abs(I_K_ideal)) * mp.atp_consumption_rate * dt
        self.metabolism.update(dt, atp_consumed, mp.blood_glucose, mp.blood_ketones)

        # ATP-dependent ion pump efficiency
        atp_factor = np.tanh(self.metabolism.get_atp() / 2.0)

        # Effective E_K shifts toward V_rest when pumps are impaired
        effective_E_K = p.E_K * atp_factor + p.V_rest * (1.0 - atp_factor)
        I_K_effective = p.g_K * (self.n ** 4) * (self.V - effective_E_K)

        # Voltage update with metabolically-modulated potassium current
        dV_dt = (I_total - (I_Na + I_K_effective + I_L)) / p.C_m
        self.V += dV_dt * dt

        # Gating variable updates
        self.m += dt * (self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m)
        self.h += dt * (self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h)
        self.n += dt * (self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n)

        # Spike detection
        spiked = False
        if self.V > 0.0 and not self.is_spiking:
            self.is_spiking = True
            spiked = True
            self.spike_times.append(t)
        elif self.V < 0.0:
            self.is_spiking = False

        return spiked


# ============================================================================
# SYNAPSE WITH METABOLISM-MODULATED PLASTICITY (Extended in Level 3)
# ============================================================================

class MetabolicPlasticSynapse:
    """
    Synapse with STP, STDP, and energy-dependent modulation.

    Extends Level 2 PlasticSynapse:
      - Synaptic transmission is scaled by presynaptic ATP (via tanh)
      - STDP weight updates are scaled by the average ATP of pre and post neurons
      - When neurons are energy-depleted, both transmission and learning degrade
    """

    def __init__(self, pre_neuron: MetabolicNeuron, params: SynapseParams):
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
        dt = t - self.last_pre_spike_time if self.last_pre_spike_time > 0 else 0.0
        self.last_pre_spike_time = t

        p = self.params

        # Decay STP factors toward baseline
        self.facilitation = 1.0 + (self.facilitation - 1.0) * np.exp(-dt / p.tau_facilitation)
        self.depression = 1.0 - (1.0 - self.depression) * np.exp(-dt / p.tau_depression)

        # Update STP
        self.facilitation += p.U_facilitation * (1.0 - self.facilitation)
        self.depression *= (1.0 - p.U_depression)

        # Schedule spike arrival
        self.spike_queue.append(t + p.delay)

    def get_current(self, t: float) -> float:
        """
        Get postsynaptic current, modulated by STP and presynaptic ATP.

        The ATP factor ensures that energy-depleted neurons transmit
        weaker signals, matching the biological reality that vesicle
        release requires ATP-dependent processes.
        """
        while self.spike_queue and self.spike_queue[0] <= t:
            self.spike_queue.popleft()
            self.pulse_end_time = t + self.params.pulse_duration

        if t < self.pulse_end_time:
            atp_factor = np.tanh(self.pre_neuron.metabolism.get_atp() / 4.0)
            effective_weight = self.weight * self.facilitation * self.depression * atp_factor
            return -effective_weight if self.params.is_inhibitory else effective_weight

        return 0.0

    def apply_stdp(self, post_neuron: MetabolicNeuron, post_spike_time: float):
        """
        Apply STDP learning rule, modulated by metabolic state.

        The weight change is scaled by the average ATP of pre and post neurons.
        This captures the biological fact that protein synthesis for long-term
        plasticity is an energy-expensive process.
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
            pre_atp = self.pre_neuron.metabolism.get_atp()
            post_atp = post_neuron.metabolism.get_atp()
            avg_atp = (pre_atp + post_atp) / 2.0
            atp_scaling = np.tanh(avg_atp / 5.0)
            delta_w *= atp_scaling

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


# ============================================================================
# NETWORK WITH METABOLISM
# ============================================================================

class MetabolicNetwork:
    """
    Network of metabolic neurons connected by energy-dependent plastic synapses.

    Extends Level 2 PlasticNetwork with:
      - MetabolicNeuron instances (ATP-coupled dynamics)
      - MetabolicPlasticSynapse connections (ATP-scaled transmission and STDP)
      - ATP level tracking for analysis
    """

    def __init__(self, sim_params: SimulationParams, metabolic_params: MetabolicParams):
        self.params = sim_params
        self.metabolic_params = metabolic_params
        neuron_params = NeuronParams()

        n_excitatory = int(sim_params.n_neurons * sim_params.excitatory_ratio)

        # Create metabolic neurons
        self.neurons: List[MetabolicNeuron] = []
        for i in range(sim_params.n_neurons):
            is_exc = i < n_excitatory
            self.neurons.append(
                MetabolicNeuron(i, is_exc, neuron_params, metabolic_params)
            )

        # Build synaptic connections
        self.synapses: Dict[int, List[MetabolicPlasticSynapse]] = {
            i: [] for i in range(sim_params.n_neurons)
        }
        self._create_connections()

        # Recording
        self.spike_history: List[Tuple[float, int]] = []
        self.weight_snapshots: List[float] = []
        self.atp_snapshots: List[float] = []

    def _create_connections(self):
        """Create random connectivity with energy-dependent synapses."""
        p = self.params
        exc_params = SynapseParams(weight=3.0, max_weight=7.5, is_inhibitory=False)
        inh_params = SynapseParams(weight=5.0, max_weight=12.5, is_inhibitory=True)

        for post_neuron in self.neurons:
            for pre_neuron in self.neurons:
                if pre_neuron.id == post_neuron.id:
                    continue
                if np.random.random() < p.connection_prob:
                    params = exc_params if pre_neuron.is_excitatory else inh_params
                    syn = MetabolicPlasticSynapse(pre_neuron, params)
                    self.synapses[post_neuron.id].append(syn)

    def get_mean_excitatory_weight(self) -> float:
        """Calculate mean weight of all excitatory synapses."""
        weights = []
        for syn_list in self.synapses.values():
            for syn in syn_list:
                if not syn.params.is_inhibitory:
                    weights.append(syn.weight)
        return float(np.mean(weights)) if weights else 0.0

    def get_mean_atp(self) -> float:
        """Calculate mean ATP level across all neurons."""
        return float(np.mean([n.metabolism.get_atp() for n in self.neurons]))

    def get_synapse_count(self) -> int:
        """Total number of synapses in the network."""
        return sum(len(s) for s in self.synapses.values())

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
                  f"ATP: {mean_atp:.3f} mM")

        print("Training complete.")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(network: MetabolicNetwork, patterns: Dict[str, np.ndarray]):
    """Visualise training results with metabolic tracking."""

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

    # --- ATP Levels ---
    ax = axes[0, 1]
    ax.plot(network.atp_snapshots, 'g-s', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean ATP (mM)')
    ax.set_title('ATP Levels Over Training')
    ax.grid(True, alpha=0.3)

    # --- Weight Distribution ---
    ax = axes[1, 0]
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
    ax = axes[1, 1]
    if network.spike_history:
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

    plt.tight_layout()
    plt.savefig('level3_metabolism_results.png', dpi=150)
    print("Results saved to: level3_metabolism_results.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=== Level 3: Bioenergetic Network ===\n")

    # Configure simulation
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

    # Configure metabolic environment
    metabolism = MetabolicParams(
        blood_glucose=5.0,
        blood_ketones=0.1,
        complex_i_efficiency=1.0,
        atp_consumption_rate=1e-5,
    )

    # Create training patterns
    patterns = {
        'A': create_sparse_pattern('A', sim.n_input_neurons, n_active=8),
        'B': create_sparse_pattern('B', sim.n_input_neurons, n_active=8),
        'C': create_sparse_pattern('C', sim.n_input_neurons, n_active=8),
    }

    print("Patterns created:")
    for name, pat in patterns.items():
        active = np.where(pat == 1)[0]
        print(f"  {name}: active neurons {active.tolist()}")
    print()

    # Create and run network
    network = MetabolicNetwork(sim, metabolism)

    initial_w = network.get_mean_excitatory_weight()
    initial_atp = network.get_mean_atp()
    print(f"Initial state: weight={initial_w:.4f}, ATP={initial_atp:.3f} mM\n")

    network.train(patterns)

    # Results
    final_w = network.get_mean_excitatory_weight()
    final_atp = network.get_mean_atp()

    print(f"\n=== Results ===")
    print(f"Initial mean excitatory weight: {initial_w:.4f}")
    print(f"Final   mean excitatory weight: {final_w:.4f}")
    print(f"Change: {final_w - initial_w:+.4f}")
    print(f"Final mean ATP: {final_atp:.3f} mM")

    if final_w > initial_w:
        print("Synaptic weights potentiated - learning occurred.")
    else:
        print("Weights did not show net potentiation.")

    # Per-neuron spike summary
    print(f"\nSpike summary (top 10 most active):")
    spike_counts = [(n.id, len(n.spike_times), 'E' if n.is_excitatory else 'I')
                    for n in network.neurons if n.spike_times]
    spike_counts.sort(key=lambda x: x[1], reverse=True)
    for nid, count, ntype in spike_counts[:10]:
        print(f"  Neuron {nid:3d} ({ntype}): {count} spikes")

    # Visualize
    plot_results(network, patterns)
