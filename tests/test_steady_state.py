"""Steady-state solutions must match the closed-form 1D geotherm."""

import numpy as np

from geothermal_sim import (
    SECONDS_PER_YEAR,
    Grid,
    HeatConductionSolver,
    Layer,
    layered_fields,
    steady_state_profile,
    uniform_fields,
)


def test_uniform_rock_matches_linear_geotherm(grid, bc):
    k = 2.5
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=k), bc)
    temperature = solver.steady_state()

    expected = bc.surface_temperature + bc.basal_heat_flux * grid.z_centers / k
    np.testing.assert_allclose(temperature, np.broadcast_to(expected[:, None], temperature.shape), rtol=0, atol=1e-9)
    # 20 °C + 0.08 W/m² * 5000 m / 2.5 W/m/K = 180 °C at the base.
    assert abs(bc.surface_temperature + bc.basal_heat_flux * grid.depth / k - 180.0) < 1e-12


def test_layered_rock_matches_piecewise_linear_geotherm(grid, bc):
    # Layer interfaces at 1 km and 3 km coincide with cell faces (dz = 100 m).
    layers = [Layer(1000.0, 1.5), Layer(3000.0, 2.5), Layer(5000.0, 3.5)]
    solver = HeatConductionSolver(grid, layered_fields(grid, layers), bc)
    temperature = solver.steady_state()

    expected = steady_state_profile(grid.z_centers, layers, bc, grid.depth)
    np.testing.assert_allclose(temperature, np.broadcast_to(expected[:, None], temperature.shape), rtol=0, atol=1e-9)

    # Temperature gradient is inversely proportional to conductivity in each layer.
    gradient = np.diff(temperature[:, 0]) / grid.dz
    assert np.isclose(gradient[:9].mean(), bc.basal_heat_flux / 1.5)
    assert np.isclose(gradient[11:29].mean(), bc.basal_heat_flux / 2.5)
    assert np.isclose(gradient[31:].mean(), bc.basal_heat_flux / 3.5)


def test_heat_production_matches_quadratic_geotherm(grid, bc):
    layers = [Layer(2000.0, 2.0, heat_production=2e-6), Layer(5000.0, 3.0, heat_production=0.5e-6)]
    solver = HeatConductionSolver(grid, layered_fields(grid, layers), bc)
    temperature = solver.steady_state()

    expected = steady_state_profile(grid.z_centers, layers, bc, grid.depth)
    # Second-order discretisation of the quadratic profile: small but non-zero error.
    assert np.abs(temperature[:, 0] - expected).max() < 0.05
    # Surface heat flow = basal flux + total radiogenic production.
    expected_surface_flux = bc.basal_heat_flux + 2e-6 * 2000.0 + 0.5e-6 * 3000.0
    np.testing.assert_allclose(solver.surface_heat_flux(temperature), expected_surface_flux, rtol=1e-9)


def test_transient_converges_to_steady_state(bc):
    grid = Grid(width=1000.0, depth=5000.0, nx=4, nz=40)
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=2.5), bc, theta=0.5)
    steady = solver.steady_state()

    # Diffusion time L²/κ ≈ 0.85 Myr; run several multiples of that.
    result = solver.run(
        solver.uniform_initial(),
        total_time=5e6 * SECONDS_PER_YEAR,
        dt=5e3 * SECONDS_PER_YEAR,
    )
    assert np.abs(result.temperature - steady).max() < 1e-3
