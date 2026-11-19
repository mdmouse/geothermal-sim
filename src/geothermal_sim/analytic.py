"""Closed-form 1D steady-state geotherms used for verification."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .materials import Layer, validate_layers
from .solver import BoundaryConditions


def steady_state_profile(
    z: np.ndarray, layers: Sequence[Layer], bc: BoundaryConditions, depth: float
) -> np.ndarray:
    """Steady 1D temperature at depths ``z`` for piecewise-constant k and Q.

    Within each layer the upward heat flux is ``q(z) = q_bottom + Q (z_bot - z)``
    and ``dT/dz = q(z) / k``; integrating layer by layer from the surface gives a
    piecewise-linear (or piecewise-quadratic, with heat production) geotherm.
    """
    z = np.asarray(z, dtype=float)
    ordered = validate_layers(layers, depth)

    bottoms = [layer.bottom for layer in ordered]
    tops = [0.0] + bottoms[:-1]

    # Heat flux at the bottom of each layer: basal flux plus everything generated below.
    q_bottom = []
    q = bc.basal_heat_flux
    for top, layer in zip(reversed(tops), reversed(ordered)):
        q_bottom.append(q)
        q += layer.heat_production * (min(layer.bottom, depth) - top)
    q_bottom.reverse()

    temperature = np.empty_like(z)
    t_top = bc.surface_temperature
    for top, layer, q_b in zip(tops, ordered, q_bottom):
        bottom = min(layer.bottom, depth)
        mask = (z >= top) & (z <= bottom)
        if np.any(mask):
            zz = z[mask]
            integral = q_b * (zz - top) + layer.heat_production * (
                bottom * (zz - top) - (zz**2 - top**2) / 2.0
            )
            temperature[mask] = t_top + integral / layer.k
        full = q_b * (bottom - top) + layer.heat_production * (
            bottom * (bottom - top) - (bottom**2 - top**2) / 2.0
        )
        t_top = t_top + full / layer.k
    return temperature
