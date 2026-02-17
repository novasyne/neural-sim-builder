"""
Example 01: Single Neuron Action Potential

Demonstrates the Hodgkin-Huxley model by injecting step current into a
single neuron and plotting the voltage trace, ion channel gating variables,
and ionic currents.

Level: Beginner
Runtime: ~5 seconds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from components.base import NeuronParams
from components.neurons import HodgkinHuxleyNeuron


def main():
    print("=== Example 01: Single Neuron Action Potential ===\n")

    params = NeuronParams()
    neuron = HodgkinHuxleyNeuron(0, True, params)

    dt = 0.01          # ms
    duration = 50.0     # ms
    n_steps = int(duration / dt)

    # Step current: off for first 5 ms, on for the rest
    step_onset = 5.0    # ms
    I_amplitude = 10.0  # uA/cm^2

    # Recording arrays
    times = np.zeros(n_steps)
    voltages = np.zeros(n_steps)
    m_gate = np.zeros(n_steps)
    h_gate = np.zeros(n_steps)
    n_gate = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt
        I_ext = I_amplitude if t >= step_onset else 0.0

        # Record state before update
        times[step] = t
        voltages[step] = neuron.V
        m_gate[step] = neuron.m
        h_gate[step] = neuron.h
        n_gate[step] = neuron.n

        neuron.step(dt, I_ext, 0.0, t)

    print(f"Simulation complete: {len(neuron.spike_times)} spikes")
    for i, spike_t in enumerate(neuron.spike_times):
        print(f"  Spike {i + 1}: t = {spike_t:.2f} ms")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Voltage trace
    ax = axes[0]
    ax.plot(times, voltages, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Threshold (~0 mV)')
    ax.axvline(x=step_onset, color='blue', linestyle=':', alpha=0.5, label='Current onset')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Hodgkin-Huxley Neuron: Action Potential')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gating variables
    ax = axes[1]
    ax.plot(times, m_gate, 'r-', label='m (Na activation)', linewidth=1.2)
    ax.plot(times, h_gate, 'b-', label='h (Na inactivation)', linewidth=1.2)
    ax.plot(times, n_gate, 'g-', label='n (K activation)', linewidth=1.2)
    ax.set_ylabel('Gate Value (0-1)')
    ax.set_title('Ion Channel Gating Variables')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ionic currents
    ax = axes[2]
    p = params
    I_Na = p.g_Na * (m_gate ** 3) * h_gate * (voltages - p.E_Na)
    I_K = p.g_K * (n_gate ** 4) * (voltages - p.E_K)
    I_L = p.g_L * (voltages - p.E_L)
    ax.plot(times, -I_Na, 'r-', label='I_Na (inward)', linewidth=1, alpha=0.8)
    ax.plot(times, -I_K, 'g-', label='I_K (outward)', linewidth=1, alpha=0.8)
    ax.plot(times, -I_L, 'b-', label='I_L (leak)', linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (uA/cm^2)')
    ax.set_title('Ionic Currents')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_single_neuron_results.png', dpi=150)
    print("\nResults saved to: 01_single_neuron_results.png")
    plt.show()


if __name__ == "__main__":
    main()
