"""Cell-centred finite-volume grid for the 2D (x, z) cross-section."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Uniform rectangular grid.

    ``x`` is horizontal (0 .. width), ``z`` is depth, positive downwards
    (0 at the surface .. depth at the bottom). Arrays are indexed ``[iz, ix]``.
    """

    width: float
    depth: float
    nx: int
    nz: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive")
        if self.nx < 1 or self.nz < 2:
            raise ValueError("need nx >= 1 and nz >= 2 cells")

    @property
    def dx(self) -> float:
        return self.width / self.nx

    @property
    def dz(self) -> float:
        return self.depth / self.nz

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def n_cells(self) -> int:
        return self.nz * self.nx

    @property
    def cell_area(self) -> float:
        """Area of one cell (per metre along strike)."""
        return self.dx * self.dz

    @property
    def x_centers(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def z_centers(self) -> np.ndarray:
        return (np.arange(self.nz) + 0.5) * self.dz

    @property
    def z_faces(self) -> np.ndarray:
        return np.arange(self.nz + 1) * self.dz
