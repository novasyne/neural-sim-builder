"""
Example 06: Layered Cortical Network

Builds a 3-layer cortical network with pyramidal neurons and basket cells,
distance-dependent connectivity, and structural plasticity. Demonstrates
feedforward processing and synaptogenesis.

Level: Advanced
Runtime: ~60 seconds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from components.base import (
    SimulationParams, MetabolicParams,
    create_sparse_pattern, create_temporal_sequence,
)
from components.networks import LayeredNetwork
from components.neurons import PyramidalNeuron, BasketCell
from components.visualization import plot_structural_results


def main():
    print("=== Example 06: Layered Cortical Network ===\n")

    sim = SimulationParams(
        dt=0.01,
        duration=100.0,
        n_neurons=128,
        n_input_neurons=64,
        excitatory_ratio=0.8,
        input_current=25.0,
        training_epochs=3,
        synaptogenesis_rate=0.0005,
    )

    metabolism = MetabolicParams(
        blood_glucose=5.0,
        blood_ketones=0.1,
        complex_i_efficiency=1.0,
    )

    np.random.seed(42)

    # Create spatiotemporal training patterns
    patterns = {}
    for label in ['A', 'B']:
        base = create_sparse_pattern(label, sim.n_input_neurons, n_active=10)
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(label))
        sequence = create_temporal_sequence(base, n_steps=3, n_flips=3, seed=seed)
        patterns[label] = sequence

    print("Patterns created:")
    for name, seq in patterns.items():
        active_counts = [int(np.sum(s)) for s in seq]
        print(f"  {name}: {len(seq)} temporal steps, "
              f"active neurons per step: {active_counts}")
    print()

    # Create network
    network = LayeredNetwork(sim, metabolism)

    # Layer summary
    for layer in sorted(set(n.layer for n in network.neurons)):
        neurons_in_layer = [n for n in network.neurons if n.layer == layer]
        n_pyr = sum(1 for n in neurons_in_layer if isinstance(n, PyramidalNeuron))
        n_bsk = sum(1 for n in neurons_in_layer if isinstance(n, BasketCell))
        print(f"  Layer {layer}: {len(neurons_in_layer)} neurons "
              f"({n_pyr} pyramidal, {n_bsk} basket)")

    initial_syn = network.get_synapse_count()
    initial_w = network.get_mean_excitatory_weight()
    print(f"\nInitial state: {initial_syn} synapses, weight={initial_w:.4f}\n")

    # Train
    network.train(patterns)

    # Results
    final_syn = network.get_synapse_count()
    final_w = network.get_mean_excitatory_weight()
    print(f"\n=== Results ===")
    print(f"Synapses: {initial_syn} -> {final_syn} ({final_syn - initial_syn:+d} new)")
    print(f"Weight: {initial_w:.4f} -> {final_w:.4f} ({final_w - initial_w:+.4f})")

    # Per-layer activity
    print(f"\nPer-layer activity:")
    for layer in sorted(set(n.layer for n in network.neurons)):
        neurons_in_layer = [n for n in network.neurons if n.layer == layer]
        total_spikes = sum(len(n.spike_times) for n in neurons_in_layer)
        avg_spikes = total_spikes / len(neurons_in_layer) if neurons_in_layer else 0
        print(f"  Layer {layer}: {total_spikes} total spikes "
              f"({avg_spikes:.1f} per neuron)")

    # Visualise
    plot_structural_results(network, save_path='06_layered_cortex_results.png')


if __name__ == "__main__":
    main()
