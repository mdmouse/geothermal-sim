"""Minimal Python-API example: layered crust, 200 kyr transient, two PNGs."""

from pathlib import Path

from geothermal_sim import (
    SECONDS_PER_YEAR,
    BoundaryConditions,
    Grid,
    HeatConductionSolver,
    Layer,
    layered_fields,
    plot_depth_profile,
    plot_heatmap,
    steady_state_profile,
)

grid = Grid(width=5000.0, depth=5000.0, nx=50, nz=100)
layers = [
    Layer(bottom=1000.0, k=2.0),  # sediments
    Layer(bottom=3000.0, k=2.5),  # upper basement
    Layer(bottom=5000.0, k=3.0),  # lower basement
]
bc = BoundaryConditions(surface_temperature=20.0, basal_heat_flux=0.080)  # 80 mW/m²
solver = HeatConductionSolver(grid, layered_fields(grid, layers), bc, theta=0.5)

result = solver.run(
    solver.uniform_initial(),
    total_time=200_000 * SECONDS_PER_YEAR,
    dt=1_000 * SECONDS_PER_YEAR,
)
print(f"bottom temperature after {result.time / SECONDS_PER_YEAR:.0f} yr: {result.temperature[-1].mean():.1f} °C")
print(f"energy balance residual: {result.energy_balance_residual / result.heat_in_bottom:.1e} (relative)")

out = Path("output")
out.mkdir(exist_ok=True)
plot_heatmap(grid, result.temperature, out / "heatmap.png")
plot_depth_profile(
    grid,
    result.temperature,
    out / "depth_profile.png",
    reference=steady_state_profile(grid.z_centers, layers, bc, grid.depth),
)
