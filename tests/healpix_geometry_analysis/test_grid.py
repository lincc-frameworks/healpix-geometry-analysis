import healpy as hp
import jax.numpy as jnp
import pytest
from healpix_geometry_analysis.grid import HealpixGrid
from numpy.testing import assert_allclose


@pytest.mark.parametrize("order", [0, 1, 2, 4, 8, 15, 17, 23])
def test_grid(order):
    """Test HealpixGrid's properties"""
    grid = HealpixGrid(order)

    assert grid.order == order
    assert grid.nside == hp.order2nside(order)
    assert grid.ntiles == hp.nside2npix(grid.nside)
    assert_allclose(grid.tile_area_steradian, hp.nside2pixarea(grid.nside))
    assert_allclose(grid.average_pixel_size_radian, hp.nside2resol(grid.nside, arcmin=False))
    assert_allclose(grid.average_pixel_size_arcmin, hp.nside2resol(grid.nside, arcmin=True))
    assert_allclose(grid.average_pixel_size_degree, hp.nside2resol(grid.nside, arcmin=False) * 180 / jnp.pi)
    assert_allclose(grid.average_pixel_size_arcsec, hp.nside2resol(grid.nside, arcmin=True) * 60)
