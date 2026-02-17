# Examples

This directory contains focused demonstration scripts showing specific neural simulation concepts.

## 📚 Available Examples

### 01_single_neuron.py
**Concept**: Action potential generation
**Level**: Beginner
**Runtime**: ~5 seconds

Demonstrates:
- Hodgkin-Huxley ion channel dynamics
- Response to step current injection
- Voltage-time traces
- Spike threshold visualization

```bash
python 01_single_neuron.py
```

**Output**: Voltage trace showing classic action potential spikes

---

### 02_network_sync.py
**Concept**: Network synchronization
**Level**: Beginner
**Runtime**: ~15 seconds

Demonstrates:
- Excitatory/inhibitory balance
- Emergent synchrony
- Population oscillations
- Spike raster patterns

```bash
python 02_network_sync.py
```

**Output**: Spike raster showing synchronized network activity

---

### 03_pattern_learning.py
**Concept**: STDP-based learning
**Level**: Intermediate
**Runtime**: ~30 seconds

Demonstrates:
- Spike-timing dependent plasticity
- Pattern presentation
- Weight evolution over epochs
- Learning curve analysis

```bash
python 03_pattern_learning.py
```

**Output**: Weight matrix and learning curves showing pattern consolidation

---

### 04_glucose_effects.py
**Concept**: Metabolic modulation
**Level**: Intermediate
**Runtime**: ~45 seconds

Demonstrates:
- ATP production from glucose
- Energy-dependent learning
- Hypoglycemia simulation
- Metabolic rescue with ketones

```bash
python 04_glucose_effects.py
```

**Output**: Comparative learning curves (normal vs low glucose)

---

### 05_ring_network.py
**Concept**: Custom network topology
**Level**: Intermediate
**Runtime**: ~20 seconds

Demonstrates:
- Directional connectivity
- Traveling wave propagation
- Topology effects on dynamics

```bash
python 05_ring_network.py
```

**Output**: Spatiotemporal activity patterns in ring topology

---

### 06_layered_cortex.py
**Concept**: Cortical architecture
**Level**: Advanced
**Runtime**: ~60 seconds

Demonstrates:
- Multi-layer organization
- Cell type diversity (pyramidal, basket cells)
- Layer-specific connectivity
- Feedforward/feedback pathways

```bash
python 06_layered_cortex.py
```

**Output**: Layer activity heatmaps and connectivity visualization

---

## 🎓 Learning Path

### For Beginners
1. `01_single_neuron.py` - Understand basic spiking
2. `02_network_sync.py` - See emergent network behavior
3. `03_pattern_learning.py` - Explore learning mechanisms

### For Intermediate Users
1. `03_pattern_learning.py` - Master STDP
2. `04_glucose_effects.py` - Add metabolic realism
3. `05_ring_network.py` - Custom topologies

### For Advanced Users
1. `06_layered_cortex.py` - Realistic architectures
2. Modify examples for your research questions
3. Combine concepts (e.g., metabolism + layers)

## 🔧 Modifying Examples

All examples follow this structure:

```python
# 1. Import components
from components.neurons import HodgkinHuxleyNeuron
from components.networks import NeuralNetwork

# 2. Set parameters
params = SimulationParams(
    n_neurons=20,
    duration=100.0,
    input_current=20.0
)

# 3. Create network
network = NeuralNetwork(params)

# 4. Run simulation
network.run()

# 5. Visualize results
plot_results(network)
```

**To customize**:
- Change parameters in step 2
- Add new analysis in step 5
- Extend network class for new features

## 📊 Expected Outputs

Each example generates:
- **Spike rasters**: When neurons fired
- **Voltage traces**: Membrane potential over time
- **Analysis plots**: Weights, learning curves, etc.
- **Console output**: Summary statistics

Plots are saved as PNG files and displayed interactively.

## 🚀 Running All Examples

```bash
# Run all examples sequentially
for example in examples/*.py; do
    echo "Running $example..."
    python "$example"
done
```

Or use the batch script:
```bash
./run_all_examples.sh
```

## 💡 Example Use Cases

### Research
- Adapt `04_glucose_effects.py` to test different metabolites
- Modify `03_pattern_learning.py` for your stimulus set
- Extend `06_layered_cortex.py` for disease models

### Education
- Use `01_single_neuron.py` in physiology class
- Demonstrate `03_pattern_learning.py` for memory lectures
- Show `02_network_sync.py` for systems neuroscience

### Exploration
- Combine examples (learning + metabolism + layers)
- Create parameter sweeps
- Test hypotheses about neural computation

## 🐛 Troubleshooting

**Example doesn't run**:
- Check dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version` (need 3.8+)

**No plots appear**:
- Try: `import matplotlib; matplotlib.use('TkAgg')`
- Check backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`

**Simulation too slow**:
- Reduce `n_neurons` or `duration`
- Install numba: `pip install numba` (10x speedup)

**Out of memory**:
- Reduce recording frequency
- Decrease network size
- Process data in chunks

## 📖 Further Reading

- **Templates**: See `templates/` for complete level implementations
- **Components**: See `components/` for building block details
- **Docs**: See `docs/` for design philosophy and guides

## 🤝 Contributing Examples

We welcome new examples! Good examples:

- Focus on one clear concept
- Run in < 2 minutes
- Include clear comments
- Produce interpretable output
- Are ~100-300 lines

Submit via pull request with:
- The example script
- Description in this README
- Expected output screenshot

---

**Questions?** Open an issue or discussion on GitHub!
