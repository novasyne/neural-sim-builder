# Neural Simulator Builder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Claude Code Skill](https://img.shields.io/badge/Claude-Skill-purple.svg)](https://claude.ai)

**A Claude Code skill for building biologically-plausible spiking neural network simulators in Python.**

Build neural simulators with progressive complexity, from basic action potentials to metabolically-aware, self-rewiring cortical networks. Perfect for students, researchers, and anyone curious about how brains compute.

## Features

- **Progressive Learning**: 4 levels from basic to advanced
- **Biologically Accurate**: Based on experimental neuroscience
- **Ready to Run**: Complete, working code in seconds
- **Rich Visualization**: Spike rasters, voltage traces, learning curves
- **Highly Customizable**: Easy to extend and modify

## What Can You Build?

### Level 1: Basic Neural Circuit
```python
# Hodgkin-Huxley neurons with realistic ion channels
network = NeuralNetwork(n_neurons=10)
network.run()
# → See action potentials in real-time!
```

### Level 2: Learning Networks
```python
# Networks that learn patterns via STDP
network.train_on_patterns(['A', 'B', 'C'], epochs=10)
# → Watch synaptic weights evolve!
```

### Level 3: Bioenergetic Networks
```python
# Metabolism-coupled neural simulation
network.set_glucose(2.5)  # Low glucose
network.train()
# → See how energy affects learning!
```

### Level 4: Structural Networks
```python
# Self-organizing cortical architecture
network = LayeredNetwork(layers=3, synaptogenesis=True)
network.train()
# → Network physically rewires itself!
```

## Installation

### As a Claude Code Skill

1. **Clone this repository**:
   ```bash
   cd ~/.claude/skills/
   git clone https://github.com/novasyne/neural-sim-builder.git
   ```

2. **Restart Claude Code** or run:
   ```bash
   claude-code reload-skills
   ```

3. **Use the skill**:
   ```
   In Claude Code: "build a neural simulator"
   ```

### Standalone Python Package

```bash
# Clone the repository
git clone https://github.com/novasyne/neural-sim-builder.git
cd neural-sim-builder

# Install dependencies
pip install numpy matplotlib

# Run an example
python templates/level1_basic.py
```

## Quick Start

### Using the Claude Skill

Simply tell Claude what you want to build:

```
You: "Build a neural network that learns patterns"

Claude: I'll create a Level 2 simulator with STDP learning!
[Generates complete working code]

You can now run:
  python simulate.py

This will show:
  - 50 neurons learning 3 patterns
  - Weight evolution via STDP
  - Spike raster plots
  - Learning curves
```

### Direct Python Usage

```python
from components.neurons import HodgkinHuxleyNeuron, NeuronParams
from components.networks import NeuralNetwork
from components.visualization import plot_results

# Configure simulation
params = SimulationParams(
    n_neurons=20,
    duration=100.0,  # ms
    input_current=20.0
)

# Create and run network
network = NeuralNetwork(params)
network.run(external_input_neurons=[0, 1, 2])

# Visualize results
plot_results(network)
```

## Examples

### Example 1: Understanding Action Potentials

```python
"""
See how a single neuron generates spikes in response to input current.
"""
from components.neurons import HodgkinHuxleyNeuron, NeuronParams
import numpy as np
import matplotlib.pyplot as plt

# Create neuron
params = NeuronParams()
neuron = HodgkinHuxleyNeuron(neuron_id=0, params=params)

# Simulate with step current
dt = 0.01
voltages = []
times = []

for t in np.arange(0, 100, dt):
    # Apply 20 µA/cm² current
    I_external = 20.0 if 10 < t < 90 else 0.0
    neuron.step(dt, I_external, I_synaptic=0.0, t=t)

    voltages.append(neuron.V)
    times.append(t)

# Plot
plt.plot(times, voltages)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Action Potentials')
plt.axhline(y=0, color='r', linestyle='--', label='Spike threshold')
plt.legend()
plt.show()
```

**Output**: Classic action potential spikes at ~50 Hz

### Example 2: Pattern Learning with STDP

```python
"""
Train a network to recognize binary patterns using STDP.
"""
from templates.level2_plasticity import *

# Create patterns (sparse binary vectors)
patterns = {
    'A': create_pattern([0, 1, 3, 7, 10]),
    'B': create_pattern([2, 4, 5, 8, 11]),
    'C': create_pattern([1, 6, 9, 12, 14])
}

# Create network with STDP
network = STDPNetwork(n_neurons=50)

# Train for 10 epochs
for epoch in range(10):
    print(f"Epoch {epoch + 1}/10")
    for name, pattern in patterns.items():
        network.present_pattern(pattern, duration=100)
        network.reset_states()

# Analyze learned weights
network.plot_weight_matrix()
network.plot_learning_curves()
```

**Output**:
- Weight matrix showing pattern-specific clusters
- Learning curve showing convergence over epochs

### Example 3: Glucose Effects on Learning

```python
"""
Compare learning under normal vs low glucose conditions.
"""
from templates.level3_metabolism import *

def run_experiment(glucose_level, name):
    """Run learning experiment with specified glucose."""
    params = MetabolicParams(
        blood_glucose=glucose_level,
        complex_i_efficiency=1.0
    )

    network = MetabolicNetwork(n_neurons=50, params=params)

    # Train on patterns
    patterns = create_patterns(n_patterns=3)
    weight_history = []

    for epoch in range(10):
        for pattern in patterns:
            network.present_pattern(pattern)
            network.reset_states()
        weight_history.append(network.get_mean_weight())

    return weight_history

# Run both conditions
normal_weights = run_experiment(5.0, "Normal Glucose")
low_weights = run_experiment(2.5, "Low Glucose")

# Plot comparison
plt.plot(normal_weights, label='Normal (5 mM)', linewidth=2)
plt.plot(low_weights, label='Low (2.5 mM)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Synaptic Weight')
plt.title('Learning Impairment Under Hypoglycemia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Output**: Learning is dramatically impaired with low glucose (~60% reduction in potentiation)

### Example 4: Self-Organizing Cortical Network

```python
"""
Watch a layered network form new connections based on activity.
"""
from templates.level4_structure import *

# Create 3-layer network with structural plasticity
network = LayeredNetwork(
    n_input=128,
    n_layer1=64,
    n_layer2=64,
    synaptogenesis_rate=0.0005
)

# Initial connectivity
initial_synapses = network.count_synapses()
print(f"Initial synapses: {initial_synapses}")

# Train on spatiotemporal patterns
patterns = create_spatiotemporal_patterns(n_patterns=3)

for epoch in range(5):
    for pattern in patterns:
        network.present_pattern(pattern)
        network.reset_states()

# Final connectivity
final_synapses = network.count_synapses()
print(f"Final synapses: {final_synapses}")
print(f"New connections formed: {final_synapses - initial_synapses}")

# Visualize network topology
network.plot_connectivity_matrix()
network.plot_layer_activity()
```

**Output**:
- Network grows ~15-20% new synapses
- Layer-specific activity patterns emerge
- Self-organized connectivity structure

## Learning Path

### For Students
1. Start with **Level 1** to understand neural signaling
2. Move to **Level 2** to see how learning works
3. Explore **Level 3** to connect metabolism and cognition
4. Challenge yourself with **Level 4** for research-level modeling

### For Researchers
1. Jump to your target level based on your question
2. Customize parameters for your specific hypothesis
3. Run parameter sweeps and statistical analysis
4. Export data for publication

### For Educators
1. Use **Level 1** for introductory neuroscience
2. Use **Level 2** for plasticity and learning courses
3. Use **Level 3** for neuroenergetics seminars
4. Use **Level 4** for advanced computational neuroscience

## Customization

### Changing Parameters

All simulations use dataclass-based configuration:

```python
@dataclass
class NeuronParams:
    C_m: float = 1.0          # Membrane capacitance (µF/cm²)
    g_Na: float = 120.0       # Sodium conductance (mS/cm²)
    g_K: float = 36.0         # Potassium conductance (mS/cm²)
    E_Na: float = 50.0        # Sodium reversal (mV)
    E_K: float = -77.0        # Potassium reversal (mV)
    # ... modify any parameter!
```

### Adding New Neuron Types

```python
class FastSpikingInterneuron(HodgkinHuxleyNeuron):
    """Parvalbumin-positive interneuron."""

    def __init__(self, neuron_id, params):
        super().__init__(neuron_id, params)
        # Faster kinetics
        self.params.g_Na = 200.0  # Higher Na conductance
        self.params.g_K = 60.0    # Higher K conductance
```

### Custom Connectivity Patterns

```python
def create_ring_network(n_neurons):
    """Connect neurons in a ring topology."""
    network = NeuralNetwork(n_neurons)

    for i in range(n_neurons):
        next_neuron = (i + 1) % n_neurons
        network.neurons[i].connect_to(
            network.neurons[next_neuron],
            weight=5.0
        )

    return network
```

## Performance Optimization

### NumPy Vectorization
```python
# Instead of loops, use NumPy arrays
voltages = np.array([neuron.V for neuron in neurons])
currents = np.dot(weight_matrix, voltages)
```

### Numba JIT Compilation
```python
from numba import jit

@jit(nopython=True)
def fast_neuron_update(V, m, h, n, I, dt):
    """10x speedup with JIT compilation."""
    # ... computation ...
    return V_new, m_new, h_new, n_new
```

### JAX GPU Acceleration
```python
import jax.numpy as jnp

# Run on GPU for 100x speedup
voltages = jnp.array(voltages)
synaptic_currents = jnp.dot(weights, voltages)
```

## Documentation

- **[upgrade_guide.md](docs/upgrade_guide.md)** - How to progress through levels

## Contributing

Contributions welcome! Areas of interest:

- New neuron models (leaky integrate-and-fire, adaptive exponential, etc.)
- Additional plasticity rules (homeostatic, metaplasticity)
- Advanced visualizations (3D network graphs, animations)
- Performance optimizations
- Tutorial notebooks
- Additional examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Based on research by:
- Hodgkin & Huxley (1952) - Action potential dynamics
- Bi & Poo (1998) - STDP experimental characterization
- Attwell & Laughlin (2001) - Brain energy consumption
- Markram et al. (2015) - Detailed cortical models

Inspired by:
- **Brian2** - Feature-rich SNN simulator
- **NEST** - Large-scale neural simulation
- **NEURON** - Compartmental modeling

## Support

- **Bug reports**: [GitHub Issues](https://github.com/novasyne/neural-sim-builder/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/novasyne/neural-sim-builder/discussions)
- **Email**: admin@novasyne.com

## Citation

If you use this in your research, please cite:

```bibtex
@software{neural_sim_builder2025,
  author = {Vos, Gideon},
  title = {Neural Simulator Builder: Progressive Biologically-Plausible Neural Network Simulation},
  year = {2025},
  url = {https://github.com/novasyne/neural-sim-builder}
}
```


[⭐ Star this repo](https://github.com/novasyne/neural-sim-builder) | [🐛 Report Bug](https://github.com/novasyne/neural-sim-builder/issues) | [💡 Request Feature](https://github.com/novasyne/neural-sim-builder/discussions)
