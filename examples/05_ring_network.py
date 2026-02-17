"""
Example 05: Ring Network Topology

Builds a ring of neurons where each neuron connects to its nearest
neighbours, creating a directional chain. Demonstrates traveling wave
propagation and topology effects on dynamics.

Level: Intermediate
Runtime: ~20 seconds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from components.base import NeuronParams, SynapseParams
from components.neurons import HodgkinHuxleyNeuron
from components.synapses import Synapse


def main():
    print("=== Example 05: Ring Network ===\n")

    n_neurons = 20
    dt = 0.01
    duration = 150.0
    n_steps = int(duration / dt)

    neuron_params = NeuronParams()
    neurons = [HodgkinHuxleyNeuron(i, True, neuron_params) for i in range(n_neurons)]

    # Build ring: each neuron connects to the next 2 neighbours (forward)
    syn_params = SynapseParams(weight=8.0, delay=2.0, pulse_duration=2.0)
    # synapse_map[post_id] = list of Synapse
    synapse_map = {i: [] for i in range(n_neurons)}

    for i in range(n_neurons):
        for offset in [1, 2]:
            post_idx = (i + offset) % n_neurons
            syn = Synapse(neurons[i], syn_params)
            synapse_map[post_idx].append(syn)

    total_syns = sum(len(s) for s in synapse_map.values())
    print(f"Ring network: {n_neurons} neurons, {total_syns} synapses")
    print(f"Topology: each neuron -> next 2 neighbours\n")

    # Stimulate neuron 0 for the first 10 ms to initiate a wave
    stim_neuron = 0
    stim_duration = 10.0
    stim_current = 20.0

    spike_history = []
    voltage_history = []

    print(f"Stimulating neuron {stim_neuron} for {stim_duration} ms...")

    for step in range(n_steps):
        t = step * dt
        spiked = []

        for neuron in neurons:
            I_ext = stim_current if (neuron.id == stim_neuron and t < stim_duration) else 0.0
            I_syn = sum(syn.get_current(t) for syn in synapse_map[neuron.id])

            if neuron.step(dt, I_ext, I_syn, t):
                spiked.append(neuron.id)
                spike_history.append((t, neuron.id))

        for nid in spiked:
            for post_id, syn_list in synapse_map.items():
                for syn in syn_list:
                    if syn.pre_neuron.id == nid:
                        syn.on_presynaptic_spike(t)

        if step % 10 == 0:
            voltage_history.append((t, [n.V for n in neurons]))

    print(f"Simulation complete: {len(spike_history)} spikes")

    # Analyse wave propagation
    first_spike_per_neuron = {}
    for t, nid in spike_history:
        if nid not in first_spike_per_neuron:
            first_spike_per_neuron[nid] = t

    if first_spike_per_neuron:
        print("\nWave propagation (first spike time per neuron):")
        for nid in range(n_neurons):
            if nid in first_spike_per_neuron:
                print(f"  Neuron {nid:2d}: {first_spike_per_neuron[nid]:.2f} ms")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Spike raster
    ax = axes[0, 0]
    if spike_history:
        times, ids = zip(*spike_history)
        ax.scatter(times, ids, s=10, c='black', marker='|')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster (Ring Network)')
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.grid(True, alpha=0.3)

    # Wave propagation timing
    ax = axes[0, 1]
    if first_spike_per_neuron:
        sorted_ids = sorted(first_spike_per_neuron.keys())
        sorted_times = [first_spike_per_neuron[nid] for nid in sorted_ids]
        ax.plot(sorted_ids, sorted_times, 'ro-', markersize=6)
    ax.set_xlabel('Neuron ID (position in ring)')
    ax.set_ylabel('First Spike Time (ms)')
    ax.set_title('Wave Propagation Delay')
    ax.grid(True, alpha=0.3)

    # Voltage heatmap
    ax = axes[1, 0]
    if voltage_history:
        rec_times, voltages = zip(*voltage_history)
        voltages = np.array(voltages)
        im = ax.imshow(voltages.T, aspect='auto', cmap='RdBu_r',
                       extent=[0, duration, n_neurons - 0.5, -0.5],
                       vmin=-80, vmax=40)
        fig.colorbar(im, ax=ax, label='mV')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Voltage Heatmap')

    # Ring topology diagram
    ax = axes[1, 1]
    angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    ax.scatter(x_pos, y_pos, s=100, c='steelblue', zorder=5)
    for i in range(n_neurons):
        ax.annotate(str(i), (x_pos[i], y_pos[i]),
                    ha='center', va='center', fontsize=7, color='white',
                    fontweight='bold')
    # Draw connection arrows for a few neurons
    for i in range(0, n_neurons, 4):
        for offset in [1]:
            j = (i + offset) % n_neurons
            ax.annotate('', xy=(x_pos[j], y_pos[j]),
                        xytext=(x_pos[i], y_pos[i]),
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        lw=1, alpha=0.5))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Ring Topology')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('05_ring_network_results.png', dpi=150)
    print("\nResults saved to: 05_ring_network_results.png")
    plt.show()


if __name__ == "__main__":
    main()
