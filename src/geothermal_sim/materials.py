"""Rock property description: horizontal layers mapped onto the grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .grid import Grid


@dataclass(frozen=True)
class Layer:
    """A horizontal rock layer extending from the previous layer's bottom to ``bottom``.

    Parameters
    ----------
    bottom : float
        Depth of the layer's lower boundary (m).
    k : float
        Thermal conductivity (W/m/K).
    rho : float
        Density (kg/m^3).
    cp : float
        Specific heat capacity (J/kg/K).
    heat_production : float
        Volumetric radiogenic heat production (W/m^3).
    """

    bottom: float
    k: float
    rho: float = 2700.0
    cp: float = 1000.0
    heat_production: float = 0.0

    def __post_init__(self) -> None:
        if self.k <= 0 or self.rho <= 0 or self.cp <= 0:
            raise ValueError("k, rho and cp must be positive")
        if self.heat_production < 0:
            raise ValueError("heat_production must be non-negative")

    @property
    def rho_cp(self) -> float:
        return self.rho * self.cp


@dataclass(frozen=True)
class MaterialFields:
    """Cell-wise material properties, each of shape ``grid.shape``."""

    k: np.ndarray
    rho_cp: np.ndarray
    heat_production: np.ndarray


def validate_layers(layers: Sequence[Layer], depth: float) -> list[Layer]:
    """Return layers sorted by bottom depth, checking they cover ``depth``."""
    if not layers:
        raise ValueError("at least one layer is required")
    ordered = sorted(layers, key=lambda layer: layer.bottom)
    if ordered[0].bottom <= 0:
        raise ValueError("layer bottoms must be positive depths")
    bottoms = [layer.bottom for layer in ordered]
    if len(set(bottoms)) != len(bottoms):
        raise ValueError("layer bottoms must be distinct")
    if ordered[-1].bottom < depth - 1e-9:
        raise ValueError(
            f"layers end at {ordered[-1].bottom} m but the domain is {depth} m deep"
        )
    return ordered


def layered_fields(grid: Grid, layers: Sequence[Layer]) -> MaterialFields:
    """Assign each cell the properties of the layer containing its centre."""
    ordered = validate_layers(layers, grid.depth)
    bottoms = np.array([layer.bottom for layer in ordered])
    layer_index = np.searchsorted(bottoms, grid.z_centers, side="left")
    layer_index = np.minimum(layer_index, len(ordered) - 1)

    def column(attr: str) -> np.ndarray:
        values = np.array([getattr(layer, attr) for layer in ordered], dtype=float)
        return np.repeat(values[layer_index][:, None], grid.nx, axis=1)

    return MaterialFields(
        k=column("k"),
        rho_cp=column("rho_cp"),
        heat_production=column("heat_production"),
    )


def uniform_fields(
    grid: Grid,
    k: float,
    rho: float = 2700.0,
    cp: float = 1000.0,
    heat_production: float = 0.0,
) -> MaterialFields:
    """Homogeneous rock: a single layer spanning the whole domain."""
    return layered_fields(grid, [Layer(grid.depth, k, rho, cp, heat_production)])
