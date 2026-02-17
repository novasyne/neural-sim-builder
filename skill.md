---
name: neural-sim-builder
description: Build biologically-plausible spiking neural network simulators in Python with progressive complexity (basic → learning → metabolism → structure)
version: 1.0.0
author: Gideon Vos
triggers:
  - "build a neural simulator"
  - "build neural simulator"
  - "create a spiking neural network"
  - "spiking neural network"
  - "neural network with plasticity"
  - "STDP learning"
  - "simulate neurons"
  - "hodgkin huxley"
  - "metabolic neural network"
  - "bioenergetic simulation"
  - "brain metabolism"
  - "cortical network"
  - "structural plasticity"
color: purple
---

# Neural Simulator Builder

I'll help you build a biologically-plausible spiking neural network simulator in Python!

This skill guides you through creating neural simulators with **four progressive levels of complexity**, based on cutting-edge computational neuroscience research.

## 🎯 What I Do

I help you build neural simulators at four sophistication levels:

### Level 1: Basic Neural Circuit ⚡
**Foundation**: Hodgkin-Huxley neurons with realistic biophysics
- Action potential generation with Na⁺, K⁺, and leak channels
- Synaptic connections with propagation delays and spatial organization
- Spike raster plots and voltage traces

**Use cases**:
- Understanding neural signaling fundamentals
- Learning computational neuroscience basics
- Teaching action potential dynamics

### Level 2: Adaptive Learning Networks 🧠
**Adds**: Synaptic plasticity and pattern learning
- **STDP** (Spike-Timing-Dependent Plasticity): Long-term learning
- **STP** (Short-Term Plasticity): Facilitation and depression
- Pattern presentation and memory formation

**Use cases**:
- Exploring Hebbian learning ("neurons that fire together, wire together")
- Pattern recognition experiments
- Understanding memory consolidation

### Level 3: Bioenergetic Networks 🔋
**Adds**: Metabolic constraints and energy coupling
- ATP production from glucose and ketones (glycolysis + OXPHOS)
- Energy-dependent ion pumps and neural excitability
- Metabolism-modulated synaptic plasticity
- Mitochondrial dysfunction models

**Use cases**:
- Studying metabolic effects on cognition
- Simulating hypoglycemia or ketogenic interventions
- Understanding neurodegenerative disease mechanisms
- Brain energetics research

### Level 4: Structural Cortical Networks 🏗️
**Adds**: Realistic architecture and physical rewiring
- Layered organization (input → processing layers)
- Diverse cell types (pyramidal cells, basket cells)
- Distance-dependent connectivity rules
- **Synaptogenesis**: Activity-dependent formation of new synapses
- Spatiotemporal pattern learning

**Use cases**:
- Modeling cortical microcircuits
- Studying network self-organization
- Realistic neural system simulations
- Advanced neuroscience research

## 🚀 How It Works

### Step 1: Tell Me Your Goal
Just describe what you want to explore:
- "I want to understand how neurons fire"
- "Show me how neural networks learn patterns"
- "Simulate the effect of low glucose on learning"
- "Build a cortical network that rewires itself"

### Step 2: I'll Recommend a Level
Based on your goal, I'll suggest the appropriate complexity level and explain what it includes.

### Step 3: I Generate Complete Code
You'll receive:
- ✅ **Main simulation script** - Ready to run immediately
- ✅ **Component library** - Clean, reusable building blocks
- ✅ **Configuration file** - Easy parameter tuning
- ✅ **Visualization code** - Matplotlib plots and analysis
- ✅ **Documentation** - Explanation of each component
- ✅ **Extension guide** - How to customize and upgrade

### Step 4: Run and Experiment
```bash
python simulate.py  # Run the simulation
# → Beautiful plots appear showing neural activity!
```

### Step 5: Customize and Extend
I'll help you:
- Adjust parameters (learning rates, network size, etc.)
- Add new features (custom neuron types, patterns)
- Debug issues
- Optimize performance
- Upgrade to the next level

## 📊 Example Interactions

**Example 1: Learning from Scratch**
```
User: "I want to see neural learning in action"

Skill: I'll create a Level 2 simulator with STDP learning!

[Generates code]

This network will:
- Have 50 neurons (40 excitatory, 10 inhibitory)
- Learn 3 binary patterns over 10 epochs
- Show weight evolution via STDP
- Plot learning curves

Run it and watch synaptic weights strengthen
for correlated activity!
```

**Example 2: Metabolism Research**
```
User: "How does glucose level affect memory formation?"

Skill: Perfect! That's a Level 3 question.

[Generates comparative simulation]

This will run two experiments:
1. Normal glucose (5 mM) - healthy learning
2. Low glucose (2.5 mM) - impaired learning

You'll see:
- ATP levels drop to ~2.5 mM (vs 5 mM)
- Synaptic potentiation reduced by ~60%
- Fewer spikes due to weak ion pumps

This models cognitive decline during hypoglycemia!
```

**Example 3: Advanced Architecture**
```
User: "Build a 3-layer cortical network"

Skill: Level 4 coming up!

[Generates layered network]

Architecture:
- Layer 0: 128 input neurons (pyramidal)
- Layer 1: 64 mixed (51 pyramidal, 13 basket)
- Layer 2: 64 mixed (processing layer)

Features:
- Distance-dependent connectivity
- Synaptogenesis during learning
- Spatiotemporal pattern sequences

Watch the network physically rewire itself!
```

## 🛠️ Technical Specifications

### Dependencies
**Minimal** (Levels 1-2):
- Python 3.8+
- NumPy
- Matplotlib

**Optional** (Levels 3-4):
- SciPy (advanced analysis)
- Seaborn (prettier plots)
- Numba (10x speedup via JIT compilation)
- JAX (100x speedup via GPU acceleration)

### Performance Targets
- 10 neurons: < 1 second
- 100 neurons: < 30 seconds
- 1000 neurons: < 5 minutes

*(On standard laptop without GPU)*

### Code Quality
- Type hints throughout
- Dataclasses for configuration
- Clear docstrings with units
- Modular, composable design
- No global state
- Extensive inline comments

### Biological Accuracy
Based on experimental neuroscience:
- Hodgkin & Huxley (1952) - Ion channel dynamics
- Bi & Poo (1998) - STDP characterization
- Attwell & Laughlin (2001) - Brain energy budget
- Markram et al. (2015) - Detailed cortical models

## 📈 Upgrade Paths

Start simple and add complexity:

```
Level 1 (Basic)
    ↓ +80 lines
Level 2 (Learning)
    ↓ +120 lines
Level 3 (Metabolism)
    ↓ +180 lines
Level 4 (Structure)
```

Each upgrade:
- Builds on previous level (no rewriting)
- Adds 1-2 new concepts
- Takes 10-15 minutes to understand
- Provides immediate visual feedback

## 💡 What I Can Help With

**During setup**:
- "Generate a Level 2 simulator"
- "What parameters should I use for pattern learning?"
- "Explain STDP"

**During experimentation**:
- "Why aren't neurons spiking?"
- "How do I add more patterns?"
- "Make the network bigger"
- "Speed up the simulation"

**For upgrades**:
- "Add metabolism to my network"
- "Convert to layered architecture"
- "Enable structural plasticity"

**For customization**:
- "Add a new neuron type"
- "Change connectivity pattern"
- "Export results to CSV"
- "Create animated visualization"

## 🎓 Educational Value

Perfect for:
- **Students**: Learning computational neuroscience
- **Researchers**: Prototyping neural models
- **Educators**: Teaching brain function
- **Engineers**: Exploring neuromorphic computing
- **Curious minds**: Understanding how brains work

## 🔬 Research Applications

Use this for:
- Metabolic effects on cognition
- Synaptic plasticity mechanisms
- Network self-organization
- Neurodegenerative disease models
- Brain-inspired AI architectures
- Parameter sensitivity analysis

## 📚 What You'll Learn

By working through the levels, you'll understand:
1. How neurons generate action potentials (ion channels)
2. How synapses transmit and modulate signals
3. How networks learn from experience (STDP)
4. How energy constrains brain function
5. How brain architecture emerges from activity
6. How to balance biological realism with computational efficiency

## 🚦 Getting Started

Just say:
- "Build a neural simulator" ← I'll ask what you want to explore
- "Create a learning network" ← I'll generate Level 2
- "Simulate brain metabolism" ← I'll create Level 3
- "I want to try the most advanced features" ← Level 4!

---

**Ready to explore the computational brain?** Tell me what you'd like to simulate! 🧠✨
