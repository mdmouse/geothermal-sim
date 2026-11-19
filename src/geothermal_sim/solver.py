"""Implicit finite-volume solver for 2D transient heat conduction.

Solves  rho*c_p * dT/dt = div(k grad T) + Q  on a rectangular cross-section
with a fixed surface temperature (Dirichlet), a prescribed basal heat flux
(Neumann) and insulated lateral sides. Time stepping uses the theta-method
(theta=0.5: Crank-Nicolson, theta=1: backward Euler) with a sparse LU
factorisation that is reused for every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Grid
from .materials import MaterialFields

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


@dataclass(frozen=True)
class BoundaryConditions:
    """Surface temperature (deg C) and basal heat flux (W/m^2, positive into the domain)."""

    surface_temperature: float = 20.0
    basal_heat_flux: float = 0.08


@dataclass
class SimulationResult:
    """Output of :meth:`HeatConductionSolver.run`.

    Heat quantities are in joules per metre along strike. ``heat_in_bottom``,
    ``heat_out_top`` and ``heat_generated`` are accumulated with the same
    theta-weighting as the time integrator, so the discrete balance
    ``energy_final - energy_initial == heat_in_bottom - heat_out_top + heat_generated``
    holds to round-off.
    """

    temperature: np.ndarray
    time: float
    dt: float
    n_steps: int
    energy_initial: float
    energy_final: float
    heat_in_bottom: float
    heat_out_top: float
    heat_generated: float
    snapshot_times: list[float] = field(default_factory=list)
    snapshots: list[np.ndarray] = field(default_factory=list)

    @property
    def energy_change(self) -> float:
        return self.energy_final - self.energy_initial

    @property
    def energy_balance_residual(self) -> float:
        """Stored-energy change minus net heat input (should be ~0)."""
        return self.energy_change - (
            self.heat_in_bottom - self.heat_out_top + self.heat_generated
        )


class HeatConductionSolver:
    def __init__(
        self,
        grid: Grid,
        materials: MaterialFields,
        bc: BoundaryConditions | None = None,
        theta: float = 0.5,
    ) -> None:
        if not 0.5 <= theta <= 1.0:
            raise ValueError("theta must lie in [0.5, 1] for unconditional stability")
        for name in ("k", "rho_cp", "heat_production"):
            if getattr(materials, name).shape != grid.shape:
                raise ValueError(f"materials.{name} must have shape {grid.shape}")
        self.grid = grid
        self.materials = materials
        self.bc = bc or BoundaryConditions()
        self.theta = theta
        self._operator, self._bc_source = self._assemble()
        self._source = self._bc_source + materials.heat_production.ravel()

    # ------------------------------------------------------------------ assembly
    def _assemble(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """Build ``A`` and ``b`` such that ``div(k grad T) ~= A @ T + b`` (W/m^3)."""
        g = self.grid
        k = self.materials.k
        nz, nx = g.shape
        n = g.n_cells
        index = np.arange(n).reshape(nz, nx)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def add_faces(a: np.ndarray, b: np.ndarray, k_a: np.ndarray, k_b: np.ndarray, h: float) -> None:
            # Harmonic-mean conductivity at the face gives exact steady-state
            # fluxes across layer interfaces lying on cell faces.
            k_face = 2.0 * k_a * k_b / (k_a + k_b)
            c = (k_face / h**2).ravel()
            a = a.ravel()
            b = b.ravel()
            rows.extend([a, a, b, b])
            cols.extend([a, b, b, a])
            vals.extend([-c, c, -c, c])

        add_faces(index[:-1, :], index[1:, :], k[:-1, :], k[1:, :], g.dz)
        if nx > 1:
            add_faces(index[:, :-1], index[:, 1:], k[:, :-1], k[:, 1:], g.dx)

        # Surface Dirichlet: conductive exchange between the first cell centre
        # (half a cell below the surface) and the fixed surface temperature.
        top = index[0, :]
        c_top = 2.0 * k[0, :] / g.dz**2
        rows.append(top)
        cols.append(top)
        vals.append(-c_top)

        b_vec = np.zeros(n)
        b_vec[top] += c_top * self.bc.surface_temperature
        # Basal Neumann: prescribed flux spread over the bottom cell volume.
        b_vec[index[-1, :]] += self.bc.basal_heat_flux / g.dz

        operator = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
        operator.sum_duplicates()
        return operator, b_vec

    # ------------------------------------------------------------- diagnostics
    def thermal_energy(self, temperature: np.ndarray) -> float:
        """Stored heat relative to 0 deg C (J per metre along strike)."""
        return float(np.sum(self.materials.rho_cp * temperature) * self.grid.cell_area)

    def top_heat_flow(self, temperature: np.ndarray) -> float:
        """Conductive heat loss through the surface (W/m, positive upward)."""
        return float(np.sum(self.surface_heat_flux(temperature)) * self.grid.dx)

    def bottom_heat_flow(self) -> float:
        """Prescribed heat input through the base (W/m)."""
        return self.bc.basal_heat_flux * self.grid.width

    def internal_heat_generation(self) -> float:
        """Total radiogenic heat production (W/m)."""
        return float(np.sum(self.materials.heat_production) * self.grid.cell_area)

    def surface_heat_flux(self, temperature: np.ndarray) -> np.ndarray:
        """Surface heat flux per unit area for each column (W/m^2)."""
        g = self.grid
        return 2.0 * self.materials.k[0, :] / g.dz * (
            temperature[0, :] - self.bc.surface_temperature
        )

    # ---------------------------------------------------------------- solving
    def steady_state(self) -> np.ndarray:
        """Solve ``div(k grad T) + Q = 0`` directly."""
        solution = spla.spsolve(self._operator.tocsc(), -self._source)
        return solution.reshape(self.grid.shape)

    def uniform_initial(self, temperature: float | None = None) -> np.ndarray:
        """Initial field at a uniform temperature (default: surface temperature)."""
        value = self.bc.surface_temperature if temperature is None else temperature
        return np.full(self.grid.shape, float(value))

    def run(
        self,
        initial_temperature: np.ndarray,
        total_time: float,
        dt: float,
        n_snapshots: int = 0,
    ) -> SimulationResult:
        """Advance ``initial_temperature`` by ``total_time`` seconds.

        ``dt`` is reduced slightly if needed so that an integer number of steps
        exactly covers ``total_time``.
        """
        if initial_temperature.shape != self.grid.shape:
            raise ValueError(f"initial_temperature must have shape {self.grid.shape}")
        if total_time <= 0 or dt <= 0:
            raise ValueError("total_time and dt must be positive")

        n_steps = int(np.ceil(total_time / dt - 1e-12))
        dt = total_time / n_steps
        theta = self.theta

        inv_rho_cp = 1.0 / self.materials.rho_cp.ravel()
        rate_operator = sp.diags(inv_rho_cp) @ self._operator
        rate_source = inv_rho_cp * self._source
        identity = sp.identity(self.grid.n_cells, format="csc")
        lhs = (identity - theta * dt * rate_operator).tocsc()
        rhs_operator = (identity + (1.0 - theta) * dt * rate_operator).tocsr()
        lu = spla.splu(lhs)

        snapshot_steps = set()
        if n_snapshots > 0:
            snapshot_steps = set(
                np.unique(np.linspace(0, n_steps, n_snapshots + 1).round().astype(int)[1:])
            )

        temperature = initial_temperature.astype(float).copy()
        vector = temperature.ravel().copy()
        energy_initial = self.thermal_energy(temperature)
        heat_in = heat_out = heat_gen = 0.0
        q_bottom = self.bottom_heat_flow()
        q_gen = self.internal_heat_generation()
        snapshot_times: list[float] = []
        snapshots: list[np.ndarray] = []

        for step in range(1, n_steps + 1):
            top_old = self.top_heat_flow(vector.reshape(self.grid.shape))
            vector = lu.solve(rhs_operator @ vector + dt * rate_source)
            temperature = vector.reshape(self.grid.shape)
            top_new = self.top_heat_flow(temperature)
            heat_in += dt * q_bottom
            heat_gen += dt * q_gen
            heat_out += dt * (theta * top_new + (1.0 - theta) * top_old)
            if step in snapshot_steps:
                snapshot_times.append(step * dt)
                snapshots.append(temperature.copy())

        return SimulationResult(
            temperature=temperature.copy(),
            time=n_steps * dt,
            dt=dt,
            n_steps=n_steps,
            energy_initial=energy_initial,
            energy_final=self.thermal_energy(temperature),
            heat_in_bottom=heat_in,
            heat_out_top=heat_out,
            heat_generated=heat_gen,
            snapshot_times=snapshot_times,
            snapshots=snapshots,
        )
