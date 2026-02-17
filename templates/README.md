# Templates

Complete, ready-to-run neural simulator implementations organized by complexity level.

Each template is a **standalone script** that demonstrates a complete neural simulation workflow from setup to visualization.

## 🎯 Level Progression

```
Level 1 (Basic)          →  Level 2 (Plasticity)  →  Level 3 (Metabolism)  →  Level 4 (Structure)
Static circuits             Learning networks         Bioenergetics            Cortical architecture
~500 lines                  ~600 lines                ~700 lines               ~800 lines
```

---

## ⚡ Level 1: Basic Neural Circuit

**File**: `level1_basic.py`
**Complexity**: ★☆☆☆
**Runtime**: ~5 seconds (10 neurons, 100 ms)
**Dependencies**: numpy, matplotlib

### What's Included
- ✅ Hodgkin-Huxley neuron model (realistic ion channels)
- ✅ Synaptic connections with propagation delays
- ✅ Random network connectivity
- ✅ External current injection
- ✅ Spike detection and recording
- ✅ Visualization (spike raster + voltage traces)

### Use This If You Want To
- Understand neural signaling fundamentals
- Learn about action potentials
- See how synapses propagate signals
- Get started with computational neuroscience

### Usage
```python
python level1_basic.py
```

### Key Parameters to Modify
```python
SimulationParams(
    n_neurons=10,           # Network size
    duration=100.0,         # Simulation time (ms)
    input_current=20.0,     # Stimulus strength (µA/cm²)
    connection_prob=0.3,    # Connectivity density
)
```

### Expected Output
- Spike raster plot showing when each neuron fired
- Voltage traces for first 3 neurons
- Console: Total spike count per neuron

### Next Steps
- Change `input_current` to see threshold effects
- Increase `n_neurons` to 50-100
- Modify `connection_prob` to change network dynamics
- **Ready for learning?** → Move to Level 2

---

## 🧠 Level 2: Adaptive Learning Networks

**File**: `level2_plasticity.py`
**Complexity**: ★★☆☆
**Runtime**: ~30 seconds (50 neurons, 10 epochs)
**Dependencies**: numpy, matplotlib

### What's Added (builds on Level 1)
- ✅ **STDP**: Spike-Timing-Dependent Plasticity (long-term learning)
- ✅ **STP**: Short-Term Plasticity (facilitation & depression)
- ✅ Pattern presentation framework
- ✅ Weight evolution tracking
- ✅ Learning curve analysis

### Use This If You Want To
- See how neural networks learn from experience
- Understand "neurons that fire together, wire together"
- Experiment with different learning rules
- Study pattern recognition

### Usage
```python
python level2_plasticity.py
```

### Key Concepts
```python
# STDP Learning Windows
LTP: Δt > 0, Δt < 20ms  → Weight increases (potentiation)
LTD: Δt < 0, |Δt| < 20ms → Weight decreases (depression)

# Short-Term Plasticity
Facilitation: Recent spikes → stronger transmission
Depression: Recent spikes → weaker transmission
```

### Expected Output
- Weight matrix showing learned connections
- Learning curves (weight evolution over epochs)
- Pattern-specific synaptic clusters
- Before/after connectivity comparison

### Next Steps
- Train on your own patterns
- Adjust STDP window size
- Tune learning rates
- **Want metabolism?** → Move to Level 3

---

## 🔋 Level 3: Bioenergetic Networks

**File**: `level3_metabolism.py`
**Complexity**: ★★★☆
**Runtime**: ~45 seconds (50 neurons, 10 epochs)
**Dependencies**: numpy, matplotlib

### What's Added (builds on Level 2)
- ✅ **Metabolic components**: ATP production/consumption
- ✅ **Glucose metabolism**: Glycolysis + oxidative phosphorylation
- ✅ **Ketone metabolism**: Alternative fuel source
- ✅ **Energy-coupled neurons**: ATP affects ion pumps
- ✅ **Metabolism-modulated learning**: STDP scales with ATP
- ✅ **ATP level tracking**: Monitor energy state over training

### Use This If You Want To
- Study brain energetics
- Understand metabolic constraints on cognition
- Simulate disease states (mitochondrial dysfunction)
- Explore dietary interventions (ketogenic diet)

### Usage
```python
python level3_metabolism.py
```

### Key Metabolic Pathways
```python
# ATP Production
Glucose → Glycolysis → 2 ATP
Glucose → OXPHOS → 28 ATP (needs mitochondria)
Ketones → OXPHOS → 28 ATP (alternative fuel)

# ATP Consumption
Ion pumps (Na+/K+-ATPase) restore gradients
Neurotransmitter synthesis and release
Protein synthesis (structural plasticity)
```

### Expected Output
- Weight evolution over training epochs
- ATP levels over time
- Final excitatory weight distribution
- Spike raster (last epoch)

### Key Parameters to Modify
```python
MetabolicParams(
    blood_glucose=5.0,          # Try lowering to simulate hypoglycemia
    blood_ketones=0.1,          # Raise to simulate ketogenic conditions
    complex_i_efficiency=1.0,   # Lower to simulate mitochondrial dysfunction
)
```

### Next Steps
- Adjust metabolic parameters to explore energy effects on learning
- Compare outcomes under different glucose/ketone levels
- Study how Complex I efficiency impacts network dynamics
- **Ready for structure?** → Move to Level 4

---

## 🏗️ Level 4: Structural Cortical Networks

**File**: `level4_structure.py`
**Complexity**: ★★★★
**Runtime**: ~60 seconds (256 neurons, 5 epochs, structural plasticity)
**Dependencies**: numpy, matplotlib

### What's Added (builds on Level 3)
- ✅ **Layered architecture**: Input → Layer 1 → Layer 2
- ✅ **Cell type diversity**: Pyramidal neurons, basket cells
- ✅ **Spatial organization**: 3D positioning
- ✅ **Distance-dependent connectivity**: Local vs long-range
- ✅ **Structural plasticity**: Synaptogenesis (new connection formation)
- ✅ **Spatiotemporal patterns**: Sequences over time
- ✅ **Network topology evolution**: Self-organization

### Use This If You Want To
- Model realistic cortical circuits
- Study network self-organization
- Understand layer-specific computations
- Research developmental neuroscience
- Build neuromorphic systems

### Usage
```python
python level4_structure.py
```

### Network Architecture
```
Layer 0 (Input):     128 pyramidal neurons (sensory input)
Layer 1 (Processing): 64 mixed (51 pyramidal, 13 basket)
Layer 2 (Integration): 64 mixed (51 pyramidal, 13 basket)

Connectivity Rules:
- L0 → L1: Strong feedforward, distance-modulated
- L1 ↔ L1: Local recurrent, strong within 150 µm
- L1 ↔ L2: Sparse long-range connections
```

### Cell Type Properties
**Pyramidal Neurons** (Excitatory):
- Long myelinated axons (8000 µm)
- Fast conduction (15 m/s)
- Glutamatergic

**Basket Cells** (Inhibitory):
- Short unmyelinated axons (2000 µm)
- Slower conduction (1 m/s)
- GABAergic
- Target cell bodies (somatic inhibition)

### Synaptogenesis Mechanism
```python
# Hebbian rule: Form new synapse if
1. Presynaptic neuron fires
2. Postsynaptic neuron fires within 20 ms
3. Random chance (formation_rate = 0.0005)

Result: Network grows ~15-20% new connections
```

### Expected Output
- Weight evolution over training epochs
- Synapse count evolution (shows synaptogenesis)
- Spike raster coloured by layer
- Layer composition and activity summary

### Key Parameters to Modify
```python
SimulationParams(
    n_neurons=256,              # Total network size
    n_input_neurons=128,        # Input layer size
    synaptogenesis_rate=0.0005, # Set to 0 to disable structural plasticity
    training_epochs=5,          # More epochs = more rewiring
)
```

### Next Steps
- Add more layers (e.g., L2/3, L4, L5, L6)
- Implement synaptic pruning
- Create feedforward/feedback loops
- Model specific cortical areas (V1, PFC, hippocampus)

---

## 🎓 Choosing Your Starting Level

### Start at Level 1 if
- New to computational neuroscience
- Want to understand basics first
- Teaching introductory course
- Building foundation knowledge

### Start at Level 2 if
- Comfortable with neural dynamics
- Interested in learning mechanisms
- Studying memory and plasticity
- Want to see adaptation

### Start at Level 3 if
- Researching brain energetics
- Studying metabolic disorders
- Interested in nutrition-cognition links
- Have metabolism background

### Start at Level 4 if
- Researching cortical circuits
- Need realistic network models
- Studying development/reorganization
- Want maximum biological detail

## 🔧 Template Customization

Each template is designed to be easily modified:

### Common Modifications
```python
# 1. Change network size
params.n_neurons = 100  # Instead of 10/50/256

# 2. Adjust simulation time
params.duration = 500.0  # Instead of 100/250 ms

# 3. Modify learning parameters
STDP_LEARNING_RATE = 0.1  # Faster learning
STDP_WINDOW = 30.0        # Wider time window

# 4. Change metabolic parameters
params.blood_glucose = 3.0    # Hypoglycemia
params.blood_ketones = 3.0    # Ketosis

# 5. Alter network topology
CONNECTION_PROB = 0.5    # Denser connectivity
```

### Adding Custom Analysis
```python
# At end of template, add:

# Export spike times
np.save('spike_times.npy', network.spike_history)

# Calculate firing rates
rates = network.calculate_firing_rates()

# Analyze synchrony
synchrony = network.measure_synchrony()

# Save figures
plt.savefig('results.pdf')
```

## 🚀 Running Templates

### Single Template
```bash
python level1_basic.py
```

### All Templates (Sequential)
```bash
for level in templates/level*.py; do
    echo "Running $level..."
    python "$level"
done
```

### With Custom Parameters
```python
# Modify directly in file, or:
python -c "
from level1_basic import *
params.n_neurons = 100
network = NeuralNetwork(params)
network.run()
plot_results(network)
"
```

## 📊 Performance Guidelines

| Level | Neurons | Time | Memory | Notes |
|-------|---------|------|--------|-------|
| 1 | 10 | 5s | 10 MB | Instant |
| 2 | 50 | 30s | 50 MB | Quick |
| 3 | 50 | 45s | 60 MB | Acceptable |
| 4 | 256 | 60s | 150 MB | Worth it! |

**Speed up**:
- Install numba: `pip install numba` → 10x faster
- Use JAX: `pip install jax` → 100x faster (GPU)

## 🐛 Troubleshooting

**Import errors**:
```python
# Templates are self-contained
# No imports from components/ needed
python level1_basic.py  # Should work immediately
```

**Slow simulation**:
```python
# Reduce network size or duration
params.n_neurons = 20  # Instead of 256
params.duration = 50.0  # Instead of 1000
```

**No visualization**:
```python
# Add before imports:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
# Then call:
plt.savefig('output.png')  # Saves instead of showing
```

## 📚 Learning Resources

After completing templates:
- **Examples**: Focused demonstrations in `examples/`
- **Components**: Reusable building blocks in `components/`
- **Docs**: Design guides in `docs/`

## 🤝 Contributing Templates

Criteria for new templates:
- Self-contained (no external imports)
- Clear progression from existing level
- Adds 1-2 major concepts
- Runs in < 2 minutes
- Includes visualization
- Well-commented

Submit via PR!

---

**Ready to start?** Pick your level and run the template! 🚀
