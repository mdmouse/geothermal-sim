"""Matplotlib output: temperature heatmap and depth profile."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .grid import Grid  # noqa: E402


def plot_heatmap(grid: Grid, temperature: np.ndarray, path: str | Path, title: str = "") -> Path:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        temperature,
        origin="upper",
        extent=(0.0, grid.width / 1000.0, grid.depth / 1000.0, 0.0),
        aspect="auto",
        cmap="inferno",
    )
    levels = np.linspace(temperature.min(), temperature.max(), 8)[1:-1]
    ax.contour(
        grid.x_centers / 1000.0,
        grid.z_centers / 1000.0,
        temperature,
        levels=levels,
        colors="white",
        linewidths=0.6,
        alpha=0.7,
    )
    fig.colorbar(image, ax=ax, label="Temperature (°C)")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title or "Temperature field")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_depth_profile(
    grid: Grid,
    temperature: np.ndarray,
    path: str | Path,
    reference: np.ndarray | None = None,
    reference_label: str = "analytic steady state",
    title: str = "",
) -> Path:
    path = Path(path)
    profile = temperature.mean(axis=1)
    depth_km = grid.z_centers / 1000.0
    fig, ax = plt.subplots(figsize=(4.5, 6))
    ax.plot(profile, depth_km, label="simulation (horizontal mean)", lw=2)
    if reference is not None:
        ax.plot(reference, depth_km, "--", label=reference_label, lw=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (km)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(title or "Temperature-depth profile")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
