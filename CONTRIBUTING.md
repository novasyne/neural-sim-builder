# Contributing to Neural Simulator Builder

Thank you for your interest in contributing! This guide will help you get started.

## 🎯 Ways to Contribute

### 1. Report Bugs 🐛
- Use the [GitHub issue tracker](https://github.com/novasyne/neural-sim-builder/issues)
- Include: Python version, OS, error message, minimal reproduction code
- Check existing issues first to avoid duplicates

### 2. Request Features 💡
- Open a [GitHub discussion](https://github.com/novasyne/neural-sim-builder/discussions)
- Describe the use case and expected behavior
- Explain why it would benefit others

### 3. Improve Documentation 📚
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve code comments
- Create Jupyter notebooks

### 4. Submit Code 🔧
- Fix bugs
- Add new features
- Optimize performance
- Improve test coverage

## 🚀 Getting Started

### Development Setup

1. **Fork and clone**:
   ```bash
   git clone https://github.com/novasyne/neural-sim-builder.git
   cd neural-sim-builder
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Run tests**:
   ```bash
   pytest tests/
   ```

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the code style guide (below)
   - Add tests for new functionality
   - Update documentation

3. **Run tests**:
   ```bash
   pytest tests/
   python -m pylint components/
   python -m mypy components/
   ```

4. **Commit**:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub.

## 📝 Code Style Guide

### Python Style
- Follow [PEP 8](https://pep8.org/)
- Use type hints (PEP 484)
- Maximum line length: 100 characters
- Use docstrings (Google style)

### Example:
```python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class NeuronParams:
    """Parameters for Hodgkin-Huxley neuron model.

    Attributes:
        C_m: Membrane capacitance in µF/cm²
        g_Na: Maximum sodium conductance in mS/cm²
    """
    C_m: float = 1.0
    g_Na: float = 120.0

def simulate_neuron(
    params: NeuronParams,
    duration: float,
    dt: float = 0.01
) -> Tuple[List[float], List[float]]:
    """Simulate a single neuron.

    Args:
        params: Neuron parameters
        duration: Simulation duration in ms
        dt: Time step in ms

    Returns:
        Tuple of (times, voltages)

    Example:
        >>> params = NeuronParams()
        >>> times, voltages = simulate_neuron(params, 100.0)
    """
    # Implementation here
    pass
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `HodgkinHuxleyNeuron`)
- Functions: `snake_case` (e.g., `calculate_update`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `SPIKE_THRESHOLD`)
- Private: `_leading_underscore` (e.g., `_internal_state`)

### Documentation
- Every public class/function needs a docstring
- Include units for physical quantities
- Provide usage examples for complex functions

## 🧪 Testing Guidelines

### Writing Tests
```python
import pytest
from components.neurons import HodgkinHuxleyNeuron, NeuronParams

def test_neuron_initialization():
    """Test that neuron initializes at resting potential."""
    params = NeuronParams(V_rest=-65.0)
    neuron = HodgkinHuxleyNeuron(0, params)
    assert neuron.V == -65.0

def test_neuron_spiking():
    """Test that neuron spikes with sufficient current."""
    params = NeuronParams()
    neuron = HodgkinHuxleyNeuron(0, params)

    spike_detected = False
    for _ in range(10000):  # 100 ms at dt=0.01
        spiked = neuron.step(dt=0.01, I_external=20.0, I_synaptic=0.0, t=0.0)
        if spiked:
            spike_detected = True
            break

    assert spike_detected, "Neuron should spike with 20 µA/cm² input"
```

### Test Coverage
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use fixtures for common setups

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=components --cov-report=html

# Specific test file
pytest tests/test_neurons.py

# Specific test
pytest tests/test_neurons.py::test_neuron_spiking
```

## 🎨 Adding New Features

### New Neuron Types

1. Inherit from base neuron class:
   ```python
   class LeakyIntegrateFireNeuron(Neuron):
       """Simplified neuron model."""

       def __init__(self, neuron_id: int, params: LIFParams):
           self.id = neuron_id
           self.V = params.V_rest
           self.threshold = params.threshold

       def step(self, dt: float, I_total: float, t: float) -> bool:
           """Update neuron state."""
           # Implement LIF dynamics
           pass
   ```

2. Add tests
3. Add example usage
4. Update documentation

### New Plasticity Rules

1. Create new synapse class:
   ```python
   class BCMSynapse(Synapse):
       """Bienenstock-Cooper-Munro learning rule."""

       def update_weight(self, pre_activity: float, post_activity: float):
           # Implement BCM rule
           pass
   ```

2. Add to `components/plasticity.py`
3. Create example showing use case
4. Add to upgrade guide

## 📋 Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows style guide
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No merge conflicts with main branch
- [ ] Commits are clear and atomic
- [ ] PR description explains what and why

## 🎯 Priority Areas

We especially welcome contributions in:

1. **New neuron models**: LIF, AdEx, Izhikevich, etc.
2. **Plasticity rules**: BCM, homeostatic, metaplasticity
3. **Visualization**: 3D networks, animations, interactive plots
4. **Performance**: Vectorization, GPU acceleration
5. **Examples**: Jupyter notebooks, tutorials
6. **Documentation**: Video tutorials, blog posts

## 💬 Communication

- **Questions**: Open a [GitHub Discussion](https://github.com/novasyne/neural-sim-builder/discussions)
- **Email**: For private matters, email admin@novasyne.com

## 📜 Code of Conduct

- Be respectful and inclusive
- Constructive criticism is welcome
- Focus on the code, not the person
- Help newcomers learn and contribute

## 🙏 Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Credited in academic citations (if desired)

Thank you for helping make neural simulation more accessible! 🧠✨
