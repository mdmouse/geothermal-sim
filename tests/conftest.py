import pytest

from geothermal_sim import BoundaryConditions, Grid


@pytest.fixture
def grid() -> Grid:
    return Grid(width=2000.0, depth=5000.0, nx=8, nz=50)


@pytest.fixture
def bc() -> BoundaryConditions:
    return BoundaryConditions(surface_temperature=20.0, basal_heat_flux=0.08)
