"""
Example 04: Glucose Effects on Learning

Compares learning under normal vs low glucose conditions using the
metabolic network. Demonstrates how ATP availability modulates STDP
and synaptic transmission.

Level: Intermediate
Runtime: ~45 seconds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from components.base import SimulationParams, MetabolicParams, create_sparse_pattern
from components.networks import MetabolicNetwork


def run_condition(label: str, sim: SimulationParams,
                  metabolic: MetabolicParams, patterns: dict):
    """Train a metabolic network and return results."""
    print(f"\n--- {label} ---")
    np.random.seed(42)
    network = MetabolicNetwork(sim, metabolic)
    network.train(patterns)
    return {
        'label': label,
        'weights': list(network.weight_snapshots),
        'atp': list(network.atp_snapshots),
        'final_weight': network.get_mean_excitatory_weight(),
        'final_atp': network.get_mean_atp(),
    }


def main():
    print("=== Example 04: Glucose Effects on Learning ===")

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

    # Create training patterns
    patterns = {
        'A': create_sparse_pattern('A', sim.n_input_neurons, n_active=6),
        'B': create_sparse_pattern('B', sim.n_input_neurons, n_active=6),
    }

    # Define conditions to compare
    conditions = [
        ("Normal Glucose (5.0 mM)", MetabolicParams(blood_glucose=5.0)),
        ("Low Glucose (2.0 mM)", MetabolicParams(blood_glucose=2.0)),
        ("Ketone Rescue (2.0 mM glu + 3.0 mM ket)",
         MetabolicParams(blood_glucose=2.0, blood_ketones=3.0)),
    ]

    results = []
    for label, metab in conditions:
        results.append(run_condition(label, sim, metab, patterns))

    # Summary
    print("\n=== Comparison ===")
    for r in results:
        print(f"  {r['label']}: "
              f"final weight={r['final_weight']:.4f}, "
              f"final ATP={r['final_atp']:.3f} mM")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50']

    # Weight evolution comparison
    ax = axes[0]
    for r, c in zip(results, colors):
        ax.plot(r['weights'], '-o', color=c, linewidth=2,
                markersize=4, label=r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Excitatory Weight')
    ax.set_title('Weight Evolution by Metabolic Condition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ATP evolution comparison
    ax = axes[1]
    for r, c in zip(results, colors):
        ax.plot(r['atp'], '-s', color=c, linewidth=2,
                markersize=4, label=r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean ATP (mM)')
    ax.set_title('ATP Levels by Metabolic Condition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('04_glucose_effects_results.png', dpi=150)
    print("\nResults saved to: 04_glucose_effects_results.png")
    plt.show()


if __name__ == "__main__":
    main()
