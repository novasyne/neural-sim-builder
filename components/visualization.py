"""
Visualization functions for each network complexity level.

Each function produces a multi-panel figure appropriate for its level:
  - plot_basic_results: Spike raster + voltage traces (Level 1)
  - plot_plasticity_results: Weight evolution + distribution + raster + overlap (Level 2)
  - plot_metabolic_results: Weight evolution + ATP + distribution + raster (Level 3)
  - plot_structural_results: Weight + synapse count + raster by layer + layer stats (Level 4)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


# ============================================================================
# LEVEL 1
# ============================================================================

def plot_basic_results(network, save_path: str = 'basic_results.png'):
    """Visualise a basic network: spike raster and voltage traces."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Spike raster
    ax = axes[0]
    if network.spike_history:
        times, neuron_ids = zip(*network.spike_history)
        ax.scatter(times, neuron_ids, s=5, c='black', marker='|')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster Plot')
    ax.set_ylim(-0.5, network.params.n_neurons - 0.5)
    ax.grid(True, alpha=0.3)

    # Voltage traces
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
    plt.savefig(save_path, dpi=150)
    print(f"Results saved to: {save_path}")
    plt.show()


# ============================================================================
# LEVEL 2
# ============================================================================

def plot_plasticity_results(network, patterns: Dict[str, np.ndarray],
                            save_path: str = 'plasticity_results.png'):
    """Visualise a plastic network: weight evolution, distribution, raster, overlap."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Weight evolution
    ax = axes[0, 0]
    ax.plot(network.weight_snapshots, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Excitatory Weight')
    ax.set_title('Weight Evolution During Training')
    ax.axhline(y=network.weight_snapshots[0], color='r', linestyle='--',
               alpha=0.5, label='Initial weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Weight distribution
    ax = axes[0, 1]
    exc_weights = [syn.weight for sl in network.synapses.values()
                   for syn in sl if not syn.params.is_inhibitory]
    ax.hist(exc_weights, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Synaptic Weight')
    ax.set_ylabel('Count')
    ax.set_title('Final Excitatory Weight Distribution')
    ax.grid(True, alpha=0.3)

    # Spike raster (last epoch)
    ax = axes[1, 0]
    if network.spike_history:
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

    # Pattern overlap
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
    plt.savefig(save_path, dpi=150)
    print(f"Results saved to: {save_path}")
    plt.show()


# ============================================================================
# LEVEL 3
# ============================================================================

def plot_metabolic_results(network, patterns: Dict[str, np.ndarray],
                           save_path: str = 'metabolic_results.png'):
    """Visualise a metabolic network: weight evolution, ATP, distribution, raster."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Weight evolution
    ax = axes[0, 0]
    ax.plot(network.weight_snapshots, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Excitatory Weight')
    ax.set_title('Weight Evolution During Training')
    ax.axhline(y=network.weight_snapshots[0], color='r', linestyle='--',
               alpha=0.5, label='Initial weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ATP levels
    ax = axes[0, 1]
    ax.plot(network.atp_snapshots, 'g-s', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean ATP (mM)')
    ax.set_title('ATP Levels Over Training')
    ax.grid(True, alpha=0.3)

    # Weight distribution
    ax = axes[1, 0]
    exc_weights = [syn.weight for sl in network.synapses.values()
                   for syn in sl if not syn.params.is_inhibitory]
    ax.hist(exc_weights, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Synaptic Weight')
    ax.set_ylabel('Count')
    ax.set_title('Final Excitatory Weight Distribution')
    ax.grid(True, alpha=0.3)

    # Spike raster (last epoch)
    ax = axes[1, 1]
    if network.spike_history:
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
    plt.savefig(save_path, dpi=150)
    print(f"Results saved to: {save_path}")
    plt.show()


# ============================================================================
# LEVEL 4
# ============================================================================

def plot_structural_results(network, save_path: str = 'structural_results.png'):
    """Visualise a layered network: weight, synapse count, raster by layer, layer stats."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Weight evolution
    ax = axes[0, 0]
    ax.plot(network.weight_snapshots, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Excitatory Weight')
    ax.set_title('Weight Evolution')
    ax.axhline(y=network.weight_snapshots[0], color='r', linestyle='--',
               alpha=0.5, label='Initial')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Synapse count
    ax = axes[0, 1]
    ax.plot(network.synapse_count_snapshots, 'm-s', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Synapses')
    ax.set_title('Synapse Count (Synaptogenesis)')
    ax.axhline(y=network.synapse_count_snapshots[0], color='r', linestyle='--',
               alpha=0.5, label='Initial')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spike raster by layer
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

    # Layer statistics
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
    plt.savefig(save_path, dpi=150)
    print(f"Results saved to: {save_path}")
    plt.show()
