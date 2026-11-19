"""Energy conservation: heat in at the base = storage change + heat out at the surface."""

import numpy as np
import pytest

from geothermal_sim import (
    SECONDS_PER_YEAR,
    HeatConductionSolver,
    Layer,
    layered_fields,
    uniform_fields,
)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_discrete_energy_balance_is_exact(grid, bc, theta):
    layers = [Layer(1500.0, 2.0, rho=2500.0, cp=900.0), Layer(5000.0, 3.0, rho=2800.0, cp=1100.0)]
    solver = HeatConductionSolver(grid, layered_fields(grid, layers), bc, theta=theta)

    result = solver.run(
        solver.uniform_initial(),
        total_time=50_000 * SECONDS_PER_YEAR,
        dt=1_000 * SECONDS_PER_YEAR,
    )

    assert result.heat_in_bottom == pytest.approx(bc.basal_heat_flux * grid.width * result.time)
    assert result.heat_out_top > 0
    assert result.energy_change > 0
    # Conservative finite-volume scheme: residual is round-off only.
    assert abs(result.energy_balance_residual) < 1e-9 * result.heat_in_bottom


def test_energy_balance_with_internal_heat_production(grid, bc):
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=2.5, heat_production=1e-6), bc)
    result = solver.run(
        solver.uniform_initial(),
        total_time=20_000 * SECONDS_PER_YEAR,
        dt=500 * SECONDS_PER_YEAR,
    )
    expected_generated = 1e-6 * grid.width * grid.depth * result.time
    assert result.heat_generated == pytest.approx(expected_generated)
    assert abs(result.energy_balance_residual) < 1e-9 * (
        result.heat_in_bottom + result.heat_generated
    )


def test_steady_state_has_no_net_heat_input(grid, bc):
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=3.0), bc)
    steady = solver.steady_state()

    # At steady state surface heat loss equals basal heat input ...
    assert solver.top_heat_flow(steady) == pytest.approx(solver.bottom_heat_flow(), rel=1e-10)

    # ... and marching forward in time changes nothing.
    result = solver.run(steady, total_time=100_000 * SECONDS_PER_YEAR, dt=10_000 * SECONDS_PER_YEAR)
    np.testing.assert_allclose(result.temperature, steady, rtol=0, atol=1e-8)
    assert abs(result.energy_change) < 1e-9 * result.heat_in_bottom
    assert result.heat_out_top == pytest.approx(result.heat_in_bottom, rel=1e-10)


def test_early_time_heat_loss_is_small(grid, bc):
    """Starting cold, almost all basal heat goes into storage before the surface warms."""
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=2.5), bc)
    result = solver.run(solver.uniform_initial(), total_time=10_000 * SECONDS_PER_YEAR, dt=100 * SECONDS_PER_YEAR)
    assert result.heat_out_top < 1e-3 * result.heat_in_bottom
    assert result.energy_change == pytest.approx(result.heat_in_bottom, rel=1e-3)
