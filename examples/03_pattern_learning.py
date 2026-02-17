"""
Example 03: STDP-Based Pattern Learning

Trains a plastic network on sparse binary patterns and tracks how
synaptic weights evolve. Demonstrates spike-timing-dependent plasticity
and short-term plasticity dynamics.

Level: Intermediate
Runtime: ~30 seconds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from components.base import SimulationParams, create_sparse_pattern
from components.networks import PlasticNetwork
from components.visualization import plot_plasticity_results


def main():
    print("=== Example 03: STDP-Based Pattern Learning ===\n")

    sim = SimulationParams(
        dt=0.01,
        duration=250.0,
        n_neurons=30,
        excitatory_ratio=0.8,
        connection_prob=0.15,
        input_current=25.0,
        n_input_neurons=20,
        training_epochs=8,
    )

    np.random.seed(42)

    # Create training patterns
    patterns = {
        'A': create_sparse_pattern('A', sim.n_input_neurons, n_active=6),
        'B': create_sparse_pattern('B', sim.n_input_neurons, n_active=6),
        'C': create_sparse_pattern('C', sim.n_input_neurons, n_active=6),
    }

    print("Patterns created:")
    for name, pat in patterns.items():
        active = np.where(pat == 1)[0]
        print(f"  {name}: active neurons {active.tolist()}")
    print()

    # Create and train network
    network = PlasticNetwork(sim)
    initial_w = network.get_mean_excitatory_weight()
    print(f"Initial mean excitatory weight: {initial_w:.4f}\n")

    network.train(patterns)

    # Results
    final_w = network.get_mean_excitatory_weight()
    print(f"\n=== Results ===")
    print(f"Initial weight: {initial_w:.4f}")
    print(f"Final   weight: {final_w:.4f}")
    print(f"Change: {final_w - initial_w:+.4f}")

    if final_w > initial_w:
        print("Synaptic weights potentiated - learning occurred.")
    else:
        print("Weights did not show net potentiation.")

    # Visualise
    plot_plasticity_results(network, patterns,
                            save_path='03_pattern_learning_results.png')


if __name__ == "__main__":
    main()
