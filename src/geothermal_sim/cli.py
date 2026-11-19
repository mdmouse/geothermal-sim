"""Command-line entry point: ``geothermal-sim`` / ``python -m geothermal_sim``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analytic import steady_state_profile
from .grid import Grid
from .materials import Layer, layered_fields
from .plotting import plot_depth_profile, plot_heatmap
from .solver import SECONDS_PER_YEAR, BoundaryConditions, HeatConductionSolver


def parse_layers(spec: str, depth: float) -> list[Layer]:
    """Parse ``"bottom:k[:rho:cp:Q],..."`` into layers (SI units, depths in m)."""
    layers = []
    for item in spec.split(","):
        parts = [float(p) for p in item.strip().split(":")]
        if len(parts) not in (2, 3, 4, 5):
            raise argparse.ArgumentTypeError(
                f"bad layer '{item}': expected bottom:k[:rho[:cp[:Q]]]"
            )
        layers.append(Layer(*parts))
    if layers[-1].bottom < depth:
        layers[-1] = Layer(
            depth, layers[-1].k, layers[-1].rho, layers[-1].cp, layers[-1].heat_production
        )
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geothermal-sim",
        description="2D geothermal heat-conduction simulator (finite volume, Crank-Nicolson).",
    )
    parser.add_argument("--width", type=float, default=5000.0, help="domain width in m (default 5000)")
    parser.add_argument("--depth", type=float, default=5000.0, help="domain depth in m (default 5000)")
    parser.add_argument("--nx", type=int, default=50, help="cells in x (default 50)")
    parser.add_argument("--nz", type=int, default=100, help="cells in z (default 100)")
    parser.add_argument(
        "--layers",
        default="5000:2.5",
        help=(
            "comma-separated layers 'bottom_m:k[:rho[:cp[:Q]]]', e.g. "
            "'1000:2.0,3000:2.5,5000:3.0' (default: uniform k=2.5 W/m/K)"
        ),
    )
    parser.add_argument("--surface-temp", type=float, default=20.0, help="surface temperature in °C")
    parser.add_argument(
        "--heat-flux", type=float, default=80.0, help="basal heat flux in mW/m² (default 80)"
    )
    parser.add_argument(
        "--years", type=float, default=200_000.0, help="simulated time in years (default 2e5)"
    )
    parser.add_argument("--dt-years", type=float, default=1000.0, help="time step in years")
    parser.add_argument(
        "--theta", type=float, default=0.5, help="time-stepping weight: 0.5 Crank-Nicolson, 1 implicit Euler"
    )
    parser.add_argument(
        "--initial",
        choices=["surface", "steady"],
        default="surface",
        help="initial field: uniform at surface temperature, or the steady state",
    )
    parser.add_argument(
        "--steady-only", action="store_true", help="skip the transient run and plot the steady state"
    )
    parser.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    grid = Grid(args.width, args.depth, args.nx, args.nz)
    layers = parse_layers(args.layers, args.depth)
    materials = layered_fields(grid, layers)
    bc = BoundaryConditions(args.surface_temp, args.heat_flux * 1e-3)
    solver = HeatConductionSolver(grid, materials, bc, theta=args.theta)

    steady = solver.steady_state()
    analytic = steady_state_profile(grid.z_centers, layers, bc, grid.depth)

    summary: dict[str, object] = {
        "grid": {"width_m": grid.width, "depth_m": grid.depth, "nx": grid.nx, "nz": grid.nz},
        "layers": [vars(layer) for layer in layers],
        "surface_temperature_C": bc.surface_temperature,
        "basal_heat_flux_mW_m2": args.heat_flux,
        "steady_state": {
            "bottom_temperature_C": float(steady[-1].mean()),
            "max_abs_error_vs_analytic_C": float(np.abs(steady.mean(axis=1) - analytic).max()),
        },
    }

    if args.steady_only:
        temperature = steady
        title = "Steady-state temperature field"
    else:
        initial = steady if args.initial == "steady" else solver.uniform_initial()
        result = solver.run(
            initial,
            total_time=args.years * SECONDS_PER_YEAR,
            dt=args.dt_years * SECONDS_PER_YEAR,
        )
        temperature = result.temperature
        title = f"Temperature after {args.years:g} years"
        summary["transient"] = {
            "years": args.years,
            "dt_years": result.dt / SECONDS_PER_YEAR,
            "n_steps": result.n_steps,
            "theta": args.theta,
            "bottom_temperature_C": float(temperature[-1].mean()),
            "mean_surface_heat_flux_mW_m2": float(
                solver.surface_heat_flux(temperature).mean() * 1e3
            ),
            "heat_in_bottom_J_per_m": result.heat_in_bottom,
            "heat_out_top_J_per_m": result.heat_out_top,
            "heat_generated_J_per_m": result.heat_generated,
            "stored_energy_change_J_per_m": result.energy_change,
            "energy_balance_residual_J_per_m": result.energy_balance_residual,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    heatmap = plot_heatmap(grid, temperature, args.out / "heatmap.png", title=title)
    profile = plot_depth_profile(
        grid, temperature, args.out / "depth_profile.png", reference=analytic, title=title
    )
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"wrote {heatmap}, {profile}, {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
