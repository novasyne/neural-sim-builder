"""
Tests for neuron models.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from templates.level1_basic import (
    HodgkinHuxleyNeuron,
    NeuronParams,
    Synapse,
    SynapseParams,
)


class TestHodgkinHuxleyNeuron:
    """Test suite for Hodgkin-Huxley neuron model."""

    def test_initialization(self):
        """Test that neuron initializes at resting potential."""
        params = NeuronParams(V_rest=-65.0)
        neuron = HodgkinHuxleyNeuron(neuron_id=0, params=params)

        assert neuron.V == -65.0, "Neuron should start at resting potential"
        assert neuron.id == 0, "Neuron ID should be set correctly"
        assert not neuron.is_spiking, "Neuron should not be spiking initially"

    def test_gating_variables_initialized(self):
        """Test that gating variables are initialized."""
        params = NeuronParams()
        neuron = HodgkinHuxleyNeuron(neuron_id=0, params=params)

        assert 0 <= neuron.m <= 1, "m gate should be between 0 and 1"
        assert 0 <= neuron.h <= 1, "h gate should be between 0 and 1"
        assert 0 <= neuron.n <= 1, "n gate should be between 0 and 1"

    def test_neuron_responds_to_input(self):
        """Test that neuron membrane potential changes with input current."""
        params = NeuronParams()
        neuron = HodgkinHuxleyNeuron(neuron_id=0, params=params)

        initial_V = neuron.V

        # Apply strong depolarizing current
        for _ in range(100):
            neuron.step(dt=0.01, I_external=20.0, I_synaptic=0.0, t=0.0)

        # Voltage should have changed
        assert neuron.V != initial_V, "Neuron should respond to input current"

    def test_neuron_generates_spike(self):
        """Test that neuron generates action potential with sufficient input."""
        params = NeuronParams()
        neuron = HodgkinHuxleyNeuron(neuron_id=0, params=params)

        spike_detected = False
        max_steps = 10000  # 100 ms at dt=0.01

        for i in range(max_steps):
            t = i * 0.01
            spiked = neuron.step(dt=0.01, I_external=20.0, I_synaptic=0.0, t=t)
            if spiked:
                spike_detected = True
                break

        assert spike_detected, "Neuron should spike with 20 µA/cm² input"
        assert len(neuron.spike_times) > 0, "Spike times should be recorded"

    def test_spike_threshold(self):
        """Test that spike is detected when crossing 0 mV."""
        params = NeuronParams()
        neuron = HodgkinHuxleyNeuron(neuron_id=0, params=params)

        spike_found = False

        for i in range(10000):
            t = i * 0.01
            spiked = neuron.step(dt=0.01, I_external=20.0, I_synaptic=0.0, t=t)

            if spiked:
                # At the moment of spike detection, voltage should be positive
                assert neuron.V > 0, "Voltage should be positive when spike is detected"
                spike_found = True
                break

        assert spike_found, "Should find a spike"

    def test_no_spike_without_input(self):
        """Test that neuron doesn't spike without input."""
        params = NeuronParams()
        neuron = HodgkinHuxleyNeuron(neuron_id=0, params=params)

        for i in range(10000):  # 100 ms
            t = i * 0.01
            spiked = neuron.step(dt=0.01, I_external=0.0, I_synaptic=0.0, t=t)
            if spiked:
                pytest.fail("Neuron should not spike without input")

        # Should stay near resting potential
        assert -70 < neuron.V < -60, "Voltage should stay near resting potential"


class TestSynapse:
    """Test suite for synaptic connections."""

    def test_synapse_initialization(self):
        """Test synapse initializes with correct parameters."""
        neuron_params = NeuronParams()
        pre_neuron = HodgkinHuxleyNeuron(0, neuron_params)

        synapse_params = SynapseParams(weight=5.0, delay=1.0)
        synapse = Synapse(pre_neuron, synapse_params)

        assert synapse.pre_neuron == pre_neuron
        assert synapse.params.weight == 5.0
        assert synapse.params.delay == 1.0

    def test_synaptic_delay(self):
        """Test that synaptic transmission is delayed."""
        neuron_params = NeuronParams()
        pre_neuron = HodgkinHuxleyNeuron(0, neuron_params)

        synapse_params = SynapseParams(weight=5.0, delay=2.0)  # 2 ms delay
        synapse = Synapse(pre_neuron, synapse_params)

        # Spike at t=0
        synapse.on_presynaptic_spike(t=0.0)

        # Should have no current immediately
        assert synapse.get_current(t=0.0) == 0.0

        # Should still have no current before delay
        assert synapse.get_current(t=1.0) == 0.0

        # Should have current after delay
        assert synapse.get_current(t=2.5) > 0.0

    def test_synaptic_current_magnitude(self):
        """Test that synaptic current matches weight."""
        neuron_params = NeuronParams()
        pre_neuron = HodgkinHuxleyNeuron(0, neuron_params)

        weight = 7.5
        synapse_params = SynapseParams(weight=weight, delay=0.0)
        synapse = Synapse(pre_neuron, synapse_params)

        synapse.on_presynaptic_spike(t=0.0)
        current = synapse.get_current(t=0.5)

        assert current == weight, "Synaptic current should equal weight"

    def test_synaptic_current_duration(self):
        """Test that synaptic current pulse has finite duration."""
        neuron_params = NeuronParams()
        pre_neuron = HodgkinHuxleyNeuron(0, neuron_params)

        synapse_params = SynapseParams(
            weight=5.0,
            delay=0.0,
            pulse_duration=2.0
        )
        synapse = Synapse(pre_neuron, synapse_params)

        synapse.on_presynaptic_spike(t=0.0)

        # Should have current during pulse
        assert synapse.get_current(t=1.0) > 0.0

        # Should have no current after pulse ends
        assert synapse.get_current(t=3.0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
