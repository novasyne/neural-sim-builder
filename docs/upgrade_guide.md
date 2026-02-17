# Progressive Upgrade Guide

How to smoothly upgrade from one level to the next by adding modular components.

## Level 1 → Level 2: Adding Plasticity

### What Changes

**Level 1**: Neurons communicate but don't learn (static weights)
**Level 2**: Synapses adapt based on spike timing (STDP learning)

### Upgrade Steps

#### Step 1: Extend Synapse with Learning

```python
# Level 1: Static Synapse
class Synapse:
    def __init__(self, pre_neuron, params):
        self.weight = params.weight

    def get_current(self, t):
        return self.weight if active else 0.0

# Level 2: Add STDP (just inherit and extend!)
class STDPSynapse(Synapse):
    """Synapse that learns via spike-timing dependent plasticity."""

    def __init__(self, pre_neuron, post_neuron, params):
        super().__init__(pre_neuron, params)
        self.post_neuron = post_neuron
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf

    def on_presynaptic_spike(self, t):
        super().on_presynaptic_spike(t)
        self.last_pre_spike = t

    def on_postsynaptic_spike(self, t):
        """Called when postsynaptic neuron fires."""
        self.last_post_spike = t
        delta_t = t - self.last_pre_spike

        # STDP learning rule
        if 0 < delta_t < 20.0:  # LTP window
            self.weight += 0.05 * np.exp(-delta_t / 20.0)
        elif -20.0 < delta_t < 0:  # LTD window
            self.weight -= 0.06 * np.exp(delta_t / 20.0)

        # Clamp weight
        self.weight = np.clip(self.weight, 0.1, 10.0)
```

#### Step 2: Minimal Network Changes

```python
# In Network class, replace:
# OLD (Level 1):
synapse = Synapse(pre_neuron, synapse_params)

# NEW (Level 2):
synapse = STDPSynapse(pre_neuron, post_neuron, synapse_params)

# In simulation loop, add ONE line:
if spiked:
    # Notify incoming synapses about postsynaptic spike
    for syn in self.synapses[neuron.id]:
        syn.on_postsynaptic_spike(t)  # <-- This is the ONLY change!
```

#### Step 3: Add Pattern Learning

```python
# New component for Level 2
class PatternPresenter:
    """Presents binary patterns to input neurons."""

    def __init__(self, pattern_size: int):
        self.patterns = {}

    def add_pattern(self, name: str, active_neurons: List[int]):
        """Register a pattern (list of neurons to activate)."""
        self.patterns[name] = active_neurons

    def get_input_current(self, pattern_name: str, neuron_id: int,
                          current_strength: float) -> float:
        """Returns current for given neuron when pattern is active."""
        if neuron_id in self.patterns[pattern_name]:
            return current_strength
        return 0.0

# Usage:
presenter = PatternPresenter(n_neurons=10)
presenter.add_pattern("A", [0, 1, 3, 7])     # Pattern A
presenter.add_pattern("B", [2, 4, 5, 8])     # Pattern B

# In training loop:
for epoch in range(10):
    for pattern_name in ["A", "B"]:
        network.run_pattern(presenter, pattern_name, duration=50)
        network.reset_neuron_states()
```

### Complete Level 2 Example

```python
# Just 3 changes to Level 1 code:

# 1. Use STDPSynapse instead of Synapse
# 2. Call on_postsynaptic_spike() when neurons fire
# 3. Add PatternPresenter for training

# That's it! Everything else stays the same.
```

---

## Level 2 → Level 3: Adding Metabolism

### What Changes

**Level 2**: Synapses learn but energy is unlimited
**Level 3**: Neurons consume ATP, metabolism affects learning

### Upgrade Steps

#### Step 1: Add Metabolic Component

```python
@dataclass
class MetabolicParams:
    """Energy metabolism parameters."""
    blood_glucose: float = 5.0      # mM
    blood_ketones: float = 0.1      # mM
    complex_i_efficiency: float = 1.0  # Mitochondrial health
    atp_consumption_rate: float = 1e-5

class MetabolicComponent:
    """Simulates ATP production from glucose and ketones."""

    def __init__(self, params: MetabolicParams):
        self.atp = 5.0  # Initial ATP (mM)
        self.params = params

    def step(self, dt: float, atp_consumed: float) -> float:
        """Update ATP levels."""
        # Glucose uptake and glycolysis
        glucose_flux = 0.01 * self.params.blood_glucose
        atp_from_glycolysis = glucose_flux * 2.0

        # Oxidative phosphorylation
        ketone_flux = 0.01 * self.params.blood_ketones
        atp_from_oxphos = (glucose_flux + ketone_flux) * 28.0 * \
                          self.params.complex_i_efficiency

        # Net change
        atp_generated = (atp_from_glycolysis + atp_from_oxphos) * dt
        self.atp += atp_generated - atp_consumed
        self.atp = max(0.01, self.atp)  # Can't go negative

        return self.atp
```

#### Step 2: Upgrade Neuron

```python
# Extend HodgkinHuxleyNeuron with metabolism
class MetabolicNeuron(HodgkinHuxleyNeuron):
    """Neuron with energy-dependent function."""

    def __init__(self, neuron_id, neuron_params, metabolic_params):
        super().__init__(neuron_id, neuron_params)
        self.metabolism = MetabolicComponent(metabolic_params)

    def step(self, dt, I_external, I_synaptic, t):
        # Calculate ionic currents (before updating)
        I_Na = self.params.g_Na * (self.m ** 3) * self.h * (self.V - self.params.E_Na)
        I_K = self.params.g_K * (self.n ** 4) * (self.V - self.params.E_K)

        # ATP consumption proportional to ion pump activity
        atp_consumed = (abs(I_Na) + abs(I_K)) * 1e-5 * dt

        # Update metabolism
        current_atp = self.metabolism.step(dt, atp_consumed)

        # ATP affects ion pump efficiency
        atp_factor = np.tanh(current_atp / 2.0)  # 0 to 1
        effective_E_K = self.params.E_K * atp_factor + \
                       self.params.V_rest * (1 - atp_factor)

        # Continue with normal HH dynamics, but use effective_E_K
        # ... (rest of step function)
```

#### Step 3: Metabolism-Aware STDP

```python
class MetabolicSTDPSynapse(STDPSynapse):
    """STDP learning scaled by ATP availability."""

    def on_postsynaptic_spike(self, t):
        # Calculate weight change as before
        delta_t = t - self.last_pre_spike
        delta_w = 0.0

        if 0 < delta_t < 20.0:
            delta_w = 0.05 * np.exp(-delta_t / 20.0)
        elif -20.0 < delta_t < 0:
            delta_w = -0.06 * np.exp(delta_t / 20.0)

        # NEW: Scale by ATP availability
        pre_atp = self.pre_neuron.metabolism.atp
        post_atp = self.post_neuron.metabolism.atp
        avg_atp = (pre_atp + post_atp) / 2.0
        atp_scaling = np.tanh(avg_atp / 5.0)  # 0 to 1

        # Apply scaled change
        self.weight += delta_w * atp_scaling
        self.weight = np.clip(self.weight, 0.1, 10.0)
```

### Level 3 Experiments

```python
# Compare learning under different metabolic conditions

# Healthy metabolism
healthy_params = MetabolicParams(
    blood_glucose=5.0,
    blood_ketones=0.1,
    complex_i_efficiency=1.0
)

# Mitochondrial dysfunction
impaired_params = MetabolicParams(
    blood_glucose=5.0,
    blood_ketones=0.1,
    complex_i_efficiency=0.6  # 40% reduction!
)

# Run both and compare learning curves
```

---

## Level 3 → Level 4: Adding Structure

### What Changes

**Level 3**: Homogeneous neurons, random connectivity
**Level 4**: Layered architecture, diverse cell types, structural plasticity

### Upgrade Steps

#### Step 1: Neuron Specialization

```python
class PyramidalNeuron(MetabolicNeuron):
    """Excitatory principal cell with long-range connections."""

    def __init__(self, neuron_id, layer, position, params):
        super().__init__(neuron_id, params.neuron, params.metabolic)
        self.layer = layer
        self.position = position
        self.is_excitatory = True
        self.axon_length = 8000.0  # μm (long!)
        self.is_myelinated = True

class BasketCell(MetabolicNeuron):
    """Inhibitory interneuron for local control."""

    def __init__(self, neuron_id, layer, position, params):
        super().__init__(neuron_id, params.neuron, params.metabolic)
        self.layer = layer
        self.position = position
        self.is_excitatory = False
        self.axon_length = 2000.0  # μm (short, local)
        self.is_myelinated = False
```

#### Step 2: Layered Network Builder

```python
class LayeredNetwork:
    """Network with cortical-like layer structure."""

    def __init__(self, params):
        self.neurons = []

        # Layer 0: Input layer (all pyramidal)
        for i in range(128):
            pos = Vector3(i % 16 - 8, i // 16 - 4, 0.0) * 20
            self.neurons.append(
                PyramidalNeuron(i, layer=0, position=pos, params=params)
            )

        # Layer 1: Mixed (80% pyramidal, 20% basket)
        neuron_id = 128
        for i in range(64):
            pos = Vector3(random(-400, 400), random(-400, 400), 200)
            if i < 51:  # 80%
                self.neurons.append(
                    PyramidalNeuron(neuron_id, layer=1, position=pos, params=params)
                )
            else:
                self.neurons.append(
                    BasketCell(neuron_id, layer=1, position=pos, params=params)
                )
            neuron_id += 1

        # Create layer-specific connectivity
        self._connect_layers()

    def _connect_layers(self):
        """Layer-specific connection rules."""
        for pre in self.neurons:
            for post in self.neurons:
                if pre.id == post.id:
                    continue

                distance = norm(pre.position, post.position)

                # L0 → L1: Strong feedforward
                if pre.layer == 0 and post.layer == 1:
                    prob = 0.4 * np.exp(-distance / 250)

                # L1 ↔ L1: Local recurrent
                elif pre.layer == 1 and post.layer == 1:
                    prob = 0.4 if distance < 150 else 0.1 * np.exp(-distance / 400)

                else:
                    prob = 0.0

                if np.random.random() < prob:
                    weight = 3.0 if pre.is_excitatory else 5.0
                    pre.connect_to(post, weight)
```

#### Step 3: Synaptogenesis (New Connections)

```python
class StructuralPlasticity:
    """Hebbian synapse formation."""

    def __init__(self, formation_rate: float = 0.0005):
        self.formation_rate = formation_rate
        self.recent_spikes = {}  # neuron_id -> last_spike_time

    def update(self, network, t, spiked_neurons):
        """Check for new synapse formation."""
        # Update spike times
        for nid in spiked_neurons:
            self.recent_spikes[nid] = t

        # For each newly spiked neuron
        for post_id in spiked_neurons:
            post_neuron = network.neurons[post_id]

            # Don't modify input layer
            if post_neuron.layer == 0:
                continue

            # Look for recent presynaptic activity
            for pre_id, pre_spike_time in self.recent_spikes.items():
                delta_t = t - pre_spike_time

                # Hebbian window: pre fired just before post
                if 0 < delta_t < 20.0:
                    # Random chance to form synapse
                    if np.random.random() < self.formation_rate:
                        pre_neuron = network.neurons[pre_id]
                        pre_neuron.connect_to(post_neuron, weight=0.2)
```

---

## Summary: Progressive Complexity

| Level | Core Addition | Lines Added | New Concepts |
|-------|---------------|-------------|--------------|
| 1 → 2 | STDP learning | ~50 | Plasticity, patterns |
| 2 → 3 | Metabolism | ~80 | ATP, energy coupling |
| 3 → 4 | Structure | ~150 | Layers, cell types, rewiring |

**Total**: ~280 lines to go from basic to advanced!

## Design Philosophy

1. **Composition over modification** - Add new classes, don't rewrite
2. **Backward compatible** - Level N code still works
3. **Minimal changes** - Each upgrade touches < 5 places
4. **Clear upgrades** - Obvious path to next level
5. **Modular** - Can mix features (e.g., STDP without metabolism)

This makes the skill easy to guide users through step-by-step!
