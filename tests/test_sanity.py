"""Basic behavioural checks, input validation and CLI smoke test."""

import numpy as np
import pytest

from geothermal_sim import (
    SECONDS_PER_YEAR,
    Grid,
    HeatConductionSolver,
    Layer,
    layered_fields,
    uniform_fields,
)
from geothermal_sim.cli import main, parse_layers


def test_grid_geometry():
    grid = Grid(width=4000.0, depth=5000.0, nx=8, nz=50)
    assert grid.dx == 500.0 and grid.dz == 100.0
    assert grid.shape == (50, 8)
    assert grid.z_centers[0] == 50.0 and grid.z_centers[-1] == 4950.0
    assert grid.z_faces[-1] == grid.depth


def test_layer_assignment_by_cell_centre():
    grid = Grid(width=1000.0, depth=5000.0, nx=2, nz=10)
    fields = layered_fields(grid, [Layer(1000.0, 1.0), Layer(5000.0, 4.0)])
    assert np.all(fields.k[:2] == 1.0) and np.all(fields.k[2:] == 4.0)


def test_invalid_inputs_are_rejected():
    grid = Grid(width=1000.0, depth=5000.0, nx=2, nz=10)
    with pytest.raises(ValueError):
        layered_fields(grid, [Layer(3000.0, 2.0)])  # does not reach the bottom
    with pytest.raises(ValueError):
        Layer(1000.0, -1.0)
    with pytest.raises(ValueError):
        HeatConductionSolver(grid, uniform_fields(grid, 2.0), theta=0.2)
    with pytest.raises(ValueError):
        Grid(width=1000.0, depth=5000.0, nx=2, nz=1)


def test_temperature_increases_with_depth_and_stays_bounded(grid, bc):
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=2.5), bc)
    steady = solver.steady_state()
    result = solver.run(solver.uniform_initial(), total_time=100_000 * SECONDS_PER_YEAR, dt=2_000 * SECONDS_PER_YEAR)
    T = result.temperature

    assert np.all(np.diff(T, axis=0) > 0)
    assert np.all(T >= bc.surface_temperature - 1e-9)
    assert np.all(T <= steady + 1e-9)
    assert T[0].mean() < steady[0].mean()
    # Horizontally uniform problem stays horizontally uniform.
    assert np.ptp(T, axis=1).max() < 1e-9


def test_implicit_scheme_is_stable_for_huge_time_steps(grid, bc):
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=2.5), bc, theta=1.0)
    result = solver.run(solver.uniform_initial(), total_time=1e7 * SECONDS_PER_YEAR, dt=1e6 * SECONDS_PER_YEAR)
    assert np.all(np.isfinite(result.temperature))
    assert np.abs(result.temperature - solver.steady_state()).max() < 1.0


def test_snapshots_are_recorded(grid, bc):
    solver = HeatConductionSolver(grid, uniform_fields(grid, k=2.5), bc)
    result = solver.run(solver.uniform_initial(), total_time=10 * SECONDS_PER_YEAR, dt=SECONDS_PER_YEAR, n_snapshots=5)
    assert len(result.snapshots) == 5
    assert result.snapshot_times[-1] == pytest.approx(result.time)
    np.testing.assert_array_equal(result.snapshots[-1], result.temperature)


def test_parse_layers_extends_last_layer_to_domain_depth():
    layers = parse_layers("1000:2.0,3000:2.5:2600:950:1e-6", depth=5000.0)
    assert [layer.bottom for layer in layers] == [1000.0, 5000.0]
    assert layers[1].rho == 2600.0 and layers[1].heat_production == 1e-6


def test_cli_writes_outputs(tmp_path):
    out = tmp_path / "run"
    code = main(
        [
            "--nx", "4", "--nz", "20", "--years", "1000", "--dt-years", "100",
            "--layers", "1000:2.0,5000:3.0", "--out", str(out),
        ]
    )
    assert code == 0
    for name in ("heatmap.png", "depth_profile.png", "summary.json"):
        assert (out / name).stat().st_size > 0
