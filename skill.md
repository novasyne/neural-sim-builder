---
name: neural-sim-builder
description: Build biologically-plausible spiking neural network simulators in Python with progressive complexity (basic → learning → metabolism → structure). All parameters validated against primary neuroscience literature.
version: 2.0.0
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

This skill guides you through creating neural simulators with **four progressive levels of complexity**, with all parameters validated against published neuroscience research.

## 🎯 What I Do

I help you build neural simulators at four sophistication levels:

### Level 1: Basic Neural Circuit ⚡
**Foundation**: Hodgkin-Huxley neurons with realistic biophysics
- Action potential generation with Na⁺, K⁺, and leak channels
- Absolute refractory period (2 ms, literature-validated)
- Synaptic connections with propagation delays
- NaN guard and input current clamping for numerical stability
- Spike raster plots and voltage traces

### Level 2: Adaptive Learning Networks 🧠
**Adds**: Synaptic plasticity and pattern learning
- **STDP** (Spike-Timing-Dependent Plasticity): LTP=0.01, LTD=0.012 per spike pair
  (Song et al. 2000; LTD > LTP for stability per Kempter et al. 1999)
- **STP** (Short-Term Plasticity): Facilitation and depression
- E:I weight ratio of 1:4 (Markram et al. 2015)

### Level 3: Bioenergetic Networks 🔋
**Adds**: Metabolic constraints, energy coupling, and homeostatic gain control
- ATP production from glucose and ketones (glycolysis + OXPHOS)
- Initial ATP: 3 mM (measured via ³¹P MRS; Zhu et al. 2012)
- **Michaelis-Menten glucose transport** via GLUT1 (Km=7 mM; Barros et al. 2005)
- **Glycogen shunt**: astrocytic energy buffer for burst activity (Dienel & Rothman 2019)
- **ATP→E_K coupling**: Na⁺/K⁺-ATPase impairment shifts K⁺ reversal (Attwell & Laughlin 2001)
- **ATP-gated AGC**: homeostatic firing-rate control that weakens under metabolic stress
  (Turrigiano & Nelson 2004)
- Mitochondrial Complex I dysfunction models

### Level 4: Structural Cortical Networks 🏗️
**Adds**: Realistic architecture and physical rewiring
- Layered organization (input → processing → integration)
- Diverse cell types (pyramidal cells, basket cells)
- Distance-dependent connectivity (Markram et al. 2015):
  feedforward 50%, local 30%, distant 5%
- **ATP-gated synaptogenesis**: new synapse formation limited by metabolic budget
  (GMR model; Rae et al. 2024)
- Spatiotemporal pattern learning

## 🚀 How It Works

### Step 1: Tell Me Your Goal
Just describe what you want to explore:
- "I want to understand how neurons fire"
- "Show me how neural networks learn patterns"
- "Simulate the effect of low glucose on learning"
- "Build a cortical network that rewires itself"

### Step 2: I'll Recommend a Level
Based on your goal, I'll suggest the appropriate complexity level.

### Step 3: I Generate Complete Code
You'll receive a ready-to-run simulation with visualizations.

### Step 4: Run and Experiment
```bash
python simulate.py  # Run the simulation
```

### Step 5: Customize and Extend
I'll help you adjust parameters, add features, debug, or upgrade to the next level.

## 🔬 Key Parameters (Literature-Validated)

| Parameter | Value | Source |
|-----------|-------|--------|
| HH conductances | g_Na=120, g_K=36, g_L=0.3 mS/cm² | Hodgkin & Huxley 1952 |
| Refractory period | 2.0 ms | CNS neurons (Gerstner et al.) |
| Initial ATP | 3.0 mM | ³¹P MRS (Zhu et al. 2012) |
| ATP consumption | 1e-4 (rate constant) | Calibrated to C++ reference |
| GLUT1 Km | 7.0 mM | Barros et al. 2005 |
| Glycogen buffer | 5.0 mM (glucose equiv.) | Dienel & Rothman 2019 |
| STDP LTP rate | 0.01 per spike pair | Song et al. 2000 |
| STDP LTD rate | 0.012 per spike pair | Kempter et al. 1999 |
| STDP window | 20 ms | Bi & Poo 1998 |
| E:I weight ratio | 1:4 | Markram et al. 2015 |
| AGC target rate | 35 Hz | Turrigiano & Nelson 2004 |
| Feedforward connectivity | 50% (peak) | Markram et al. 2015 |
| Local connectivity | 30% (<150 µm) | Lefort et al. 2009 |

## 🛠️ Technical Specifications

### Dependencies
**Minimal** (Levels 1-2): Python 3.8+, NumPy, Matplotlib
**Optional** (Levels 3-4): SciPy, Seaborn, Numba, JAX

### New in v2.0
- ✅ Absolute refractory period (all levels)
- ✅ NaN guard and input current clamping (all levels)
- ✅ Corrected ATP to 3 mM from ³¹P MRS data
- ✅ Michaelis-Menten glucose transport (supply constraints)
- ✅ Glycogen shunt (astrocytic energy buffer)
- ✅ ATP-gated AGC (homeostatic plasticity)
- ✅ Corrected STDP rates (0.01/0.012 from literature)
- ✅ E:I weight ratio 1:4 (corrected from 3:5)
- ✅ ATP-gated synaptogenesis (GMR energy budget)
- ✅ Corrected connection probabilities (0.5/0.3/0.05)
- ✅ Presynaptic-only ATP gating for STDP
- ✅ Gain history tracking and visualization

### Code Quality
- Type hints throughout
- Dataclasses for configuration with literature citations
- Clear docstrings with biological units
- Modular, composable design
- No global state

## 📚 Literature References

- Hodgkin & Huxley (1952) — Ion channel dynamics
- Bi & Poo (1998) — STDP characterization
- Song et al. (2000) — STDP computational rates
- Kempter et al. (1999) — LTD > LTP stability requirement
- Attwell & Laughlin (2001) — Brain energy budget
- Turrigiano & Nelson (2004) — Homeostatic plasticity
- Barros et al. (2005) — GLUT1 transport kinetics
- Du et al. (2008) — Brain ATP via ³¹P MRS
- Howarth et al. (2012) — Updated energy budget
- Zhu et al. (2012) — Brain ATP concentration (~3 mM)
- Markram et al. (2015) — Cortical microcircuit connectivity
- Dienel & Rothman (2019) — Glycogen shunt
- Rae et al. (2024) — Brain energy constraints and allostatic load

## 🚦 Getting Started

Just say:
- "Build a neural simulator" ← I'll ask what you want to explore
- "Create a learning network" ← I'll generate Level 2
- "Simulate brain metabolism" ← I'll create Level 3
- "I want the most advanced features" ← Level 4!

---

**Ready to explore the computational brain?** Tell me what you'd like to simulate! 🧠✨
