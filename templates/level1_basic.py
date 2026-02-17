"""
Level 1: Basic Neural Circuit
A minimal, clean implementation of spiking neurons with Hodgkin-Huxley dynamics.

This example shows the foundational pattern all higher levels build upon.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================================
# PARAMETERS (Clean, Type-Safe Configuration)
# ============================================================================

@dataclass
class NeuronParams:
    """Hodgkin-Huxley neuron parameters with biological units."""
    # Membrane properties
    C_m: float = 1.0          # Membrane capacitance (μF/cm²)
    V_rest: float = -65.0     # Resting potential (mV)

    # Conductances (mS/cm²)
    g_Na: float = 120.0       # Sodium max conductance
    g_K: float = 36.0         # Potassium max conductance
    g_L: float = 0.3          # Leak conductance

    # Reversal potentials (mV)
    E_Na: float = 50.0        # Sodium
    E_K: float = -77.0        # Potassium
    E_L: float = -54.387      # Leak


@dataclass
class SynapseParams:
    """Synaptic connection parameters."""
    weight: float = 5.0           # Synaptic strength
    delay: float = 1.0            # Propagation delay (ms)
    pulse_duration: float = 2.0   # Postsynaptic current duration (ms)


@dataclass
class SimulationParams:
    """Global simulation parameters."""
    dt: float = 0.01              # Time step (ms)
    duration: float = 100.0       # Total simulation time (ms)
    n_neurons: int = 10           # Number of neurons
    connection_prob: float = 0.3  # Random connectivity probability
    input_current: float = 20.0   # External input current


# ============================================================================
# NEURON (Core Component)
# ============================================================================

class HodgkinHuxleyNeuron:
    """
    Biologically realistic neuron using Hodgkin-Huxley model.

    Simulates action potentials through voltage-gated ion channels.
    """

    def __init__(self, neuron_id: int, params: NeuronParams):
        self.id = neuron_id
        self.params = params

        # State variables
        self.V = params.V_rest      # Membrane potential
        self.m = 0.05               # Na activation gate
        self.h = 0.6                # Na inactivation gate
        self.n = 0.32               # K activation gate

        # Spike detection
        self.is_spiking = False
        self.spike_times = []

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

        Returns:
            bool: True if neuron spiked this step
        """
        p = self.params
        I_total = I_external + I_synaptic

        # Calculate ionic currents
        I_Na = p.g_Na * (self.m ** 3) * self.h * (self.V - p.E_Na)
        I_K = p.g_K * (self.n ** 4) * (self.V - p.E_K)
        I_L = p.g_L * (self.V - p.E_L)

        # Update voltage
        dV_dt = (I_total - I_Na - I_K - I_L) / p.C_m
        self.V += dV_dt * dt

        # Update gating variables
        self.m += dt * (self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m)
        self.h += dt * (self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h)
        self.n += dt * (self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n)

        # Detect spike (threshold crossing at 0 mV)
        spiked = False
        if self.V > 0.0 and not self.is_spiking:
            self.is_spiking = True
            spiked = True
            self.spike_times.append(t)
        elif self.V < 0.0:
            self.is_spiking = False

        return spiked


# ============================================================================
# SYNAPSE (Connection Component)
# ============================================================================

class Synapse:
    """
    Simple synaptic connection with delay and temporal dynamics.
    """

    def __init__(self, pre_neuron: HodgkinHuxleyNeuron, params: SynapseParams):
        self.pre_neuron = pre_neuron
        self.params = params
        self.spike_queue = []  # (arrival_time, weight)
        self.pulse_end_time = -np.inf

    def on_presynaptic_spike(self, t: float):
        """Called when presynaptic neuron fires."""
        arrival_time = t + self.params.delay
        self.spike_queue.append(arrival_time)

    def get_current(self, t: float) -> float:
        """Get postsynaptic current at time t."""
        # Check for arriving spikes
        while self.spike_queue and self.spike_queue[0] <= t:
            self.spike_queue.pop(0)
            self.pulse_end_time = t + self.params.pulse_duration

        # Return current if pulse is active
        if t < self.pulse_end_time:
            return self.params.weight
        return 0.0


# ============================================================================
# NETWORK (Orchestrator)
# ============================================================================

class NeuralNetwork:
    """
    Network of Hodgkin-Huxley neurons with synaptic connections.
    """

    def __init__(self, sim_params: SimulationParams):
        self.params = sim_params

        # Create neurons
        neuron_params = NeuronParams()
        self.neurons = [
            HodgkinHuxleyNeuron(i, neuron_params)
            for i in range(sim_params.n_neurons)
        ]

        # Create random connectivity
        self.synapses = {}  # {post_id: [list of synapses]}
        self._create_random_connections()

        # Recording
        self.voltage_history = []
        self.spike_history = []  # (time, neuron_id)

    def _create_random_connections(self):
        """Create random synaptic connections."""
        synapse_params = SynapseParams()

        for post_idx, post_neuron in enumerate(self.neurons):
            self.synapses[post_idx] = []

            for pre_neuron in self.neurons:
                if pre_neuron.id == post_neuron.id:
                    continue  # No self-connections

                if np.random.random() < self.params.connection_prob:
                    synapse = Synapse(pre_neuron, synapse_params)
                    self.synapses[post_idx].append(synapse)

    def run(self, external_input_neurons: Optional[List[int]] = None):
        """
        Run the simulation.

        Args:
            external_input_neurons: List of neuron IDs to receive external current
        """
        if external_input_neurons is None:
            external_input_neurons = [0, 1]  # Default: stimulate first 2 neurons

        dt = self.params.dt
        n_steps = int(self.params.duration / dt)

        print(f"Running simulation for {self.params.duration} ms...")
        print(f"Network: {self.params.n_neurons} neurons, "
              f"{sum(len(syns) for syns in self.synapses.values())} synapses")

        for step in range(n_steps):
            t = step * dt

            # Update each neuron
            voltages = []
            for neuron in self.neurons:
                # External input
                I_ext = self.params.input_current if neuron.id in external_input_neurons else 0.0

                # Synaptic input
                I_syn = sum(
                    syn.get_current(t)
                    for syn in self.synapses.get(neuron.id, [])
                )

                # Update neuron
                spiked = neuron.step(dt, I_ext, I_syn, t)

                # Record spike
                if spiked:
                    self.spike_history.append((t, neuron.id))
                    # Notify downstream synapses
                    for post_id in range(len(self.neurons)):
                        for syn in self.synapses[post_id]:
                            if syn.pre_neuron.id == neuron.id:
                                syn.on_presynaptic_spike(t)

                voltages.append(neuron.V)

            # Record voltages (every 10 steps to save memory)
            if step % 10 == 0:
                self.voltage_history.append((t, voltages.copy()))

        print(f"Simulation complete! Total spikes: {len(self.spike_history)}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(network: NeuralNetwork):
    """Create comprehensive visualization of simulation results."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Subplot 1: Spike Raster ---
    ax = axes[0]
    if network.spike_history:
        times, neuron_ids = zip(*network.spike_history)
        ax.scatter(times, neuron_ids, s=5, c='black', marker='|')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster Plot')
    ax.set_ylim(-0.5, network.params.n_neurons - 0.5)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Voltage Traces (first 3 neurons) ---
    ax = axes[1]
    if network.voltage_history:
        times, voltages = zip(*network.voltage_history)
        times = np.array(times)
        voltages = np.array(voltages)

        for neuron_id in range(min(3, network.params.n_neurons)):
            ax.plot(times, voltages[:, neuron_id],
                   label=f'Neuron {neuron_id}', linewidth=1)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Voltage Traces (First 3 Neurons)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neural_simulation_results.png', dpi=150)
    print("Results saved to: neural_simulation_results.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configure simulation
    params = SimulationParams(
        dt=0.01,
        duration=100.0,
        n_neurons=10,
        connection_prob=0.3,
        input_current=20.0
    )

    # Create and run network
    network = NeuralNetwork(params)
    network.run(external_input_neurons=[0, 1, 2])

    # Visualize results
    plot_results(network)

    # Analysis
    print("\n=== Analysis ===")
    for neuron in network.neurons:
        print(f"Neuron {neuron.id}: {len(neuron.spike_times)} spikes")
