"""geothermal_sim: 2D finite-volume geothermal heat-conduction simulator."""

from .analytic import steady_state_profile
from .grid import Grid
from .materials import Layer, MaterialFields, layered_fields, uniform_fields
from .plotting import plot_depth_profile, plot_heatmap
from .solver import (
    SECONDS_PER_YEAR,
    BoundaryConditions,
    HeatConductionSolver,
    SimulationResult,
)

__all__ = [
    "SECONDS_PER_YEAR",
    "BoundaryConditions",
    "Grid",
    "HeatConductionSolver",
    "Layer",
    "MaterialFields",
    "SimulationResult",
    "layered_fields",
    "plot_depth_profile",
    "plot_heatmap",
    "steady_state_profile",
    "uniform_fields",
]

__version__ = "0.1.0"
