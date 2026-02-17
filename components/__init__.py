"""
Neural Simulator Components

Modular building blocks for creating biologically-plausible neural network simulators.
Import individual classes or use the full-template levels directly.
"""

# Base types and utilities
from .base import (
    Vector3,
    NeuronParams,
    MetabolicParams,
    SynapseParams,
    SimulationParams,
    STDP_WINDOW,
    create_sparse_pattern,
    create_temporal_sequence,
)

# Neuron models (progressive complexity)
from .neurons import (
    HodgkinHuxleyNeuron,
    MetabolicComponent,
    MetabolicNeuron,
    StructuralNeuron,
    PyramidalNeuron,
    BasketCell,
)

# Synapse models (progressive complexity)
from .synapses import (
    Synapse,
    PlasticSynapse,
    MetabolicPlasticSynapse,
    StructuralSynapse,
)

# Network orchestrators (progressive complexity)
from .networks import (
    NeuralNetwork,
    PlasticNetwork,
    MetabolicNetwork,
    LayeredNetwork,
)

# Visualization
from .visualization import (
    plot_basic_results,
    plot_plasticity_results,
    plot_metabolic_results,
    plot_structural_results,
)

__version__ = "1.0.0"
__all__ = [
    # Base
    "Vector3", "NeuronParams", "MetabolicParams", "SynapseParams",
    "SimulationParams", "STDP_WINDOW",
    "create_sparse_pattern", "create_temporal_sequence",
    # Neurons
    "HodgkinHuxleyNeuron", "MetabolicComponent", "MetabolicNeuron",
    "StructuralNeuron", "PyramidalNeuron", "BasketCell",
    # Synapses
    "Synapse", "PlasticSynapse", "MetabolicPlasticSynapse", "StructuralSynapse",
    # Networks
    "NeuralNetwork", "PlasticNetwork", "MetabolicNetwork", "LayeredNetwork",
    # Visualization
    "plot_basic_results", "plot_plasticity_results",
    "plot_metabolic_results", "plot_structural_results",
]
