"""
Example 02: Network Synchronization

Creates a small network of excitatory and inhibitory neurons and
observes emergent synchronization patterns. Demonstrates how
excitatory/inhibitory balance shapes population dynamics.

Level: Beginner
Runtime: ~15 seconds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from components.base import SimulationParams
from components.networks import NeuralNetwork


def main():
    print("=== Example 02: Network Synchronization ===\n")

    params = SimulationParams(
        dt=0.01,
        duration=200.0,
        n_neurons=20,
        connection_prob=0.3,
        input_current=15.0,
    )

    np.random.seed(42)
    network = NeuralNetwork(params)

    # Stimulate a subset of neurons
    input_neurons = list(range(5))
    print(f"Stimulating neurons: {input_neurons}")
    network.run(external_input_neurons=input_neurons)

    # Analyse synchronization
    if network.spike_history:
        times, ids = zip(*network.spike_history)
        times = np.array(times)

        # Compute population firing rate in 5 ms bins
        bin_width = 5.0
        bins = np.arange(0, params.duration + bin_width, bin_width)
        pop_rate, bin_edges = np.histogram(times, bins=bins)
        pop_rate = pop_rate / (bin_width / 1000.0) / params.n_neurons  # Hz

        # Per-neuron spike counts
        spike_counts = np.zeros(params.n_neurons)
        for t, nid in network.spike_history:
            spike_counts[nid] += 1
    else:
        pop_rate = np.array([])
        bin_edges = np.array([])
        spike_counts = np.zeros(params.n_neurons)

    # Print summary
    active = np.sum(spike_counts > 0)
    print(f"\nActive neurons: {active}/{params.n_neurons}")
    print(f"Total spikes: {int(np.sum(spike_counts))}")
    if len(pop_rate) > 0:
        print(f"Peak population rate: {np.max(pop_rate):.1f} Hz")
        print(f"Mean population rate: {np.mean(pop_rate):.1f} Hz")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Spike raster
    ax = axes[0]
    if network.spike_history:
        ax.scatter(times, ids, s=8, c='black', marker='|')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster')
    ax.set_ylim(-0.5, params.n_neurons - 0.5)
    ax.grid(True, alpha=0.3)

    # Population firing rate
    ax = axes[1]
    if len(pop_rate) > 0:
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centres, pop_rate, width=bin_width * 0.9,
               color='steelblue', alpha=0.7)
    ax.set_ylabel('Population Rate (Hz)')
    ax.set_title('Population Firing Rate')
    ax.grid(True, alpha=0.3)

    # Voltage traces (first 3 neurons)
    ax = axes[2]
    if network.voltage_history:
        rec_times, voltages = zip(*network.voltage_history)
        rec_times = np.array(rec_times)
        voltages = np.array(voltages)
        for nid in range(min(3, params.n_neurons)):
            ax.plot(rec_times, voltages[:, nid],
                    label=f'Neuron {nid}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Voltage Traces')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('02_network_sync_results.png', dpi=150)
    print("\nResults saved to: 02_network_sync_results.png")
    plt.show()


if __name__ == "__main__":
    main()
