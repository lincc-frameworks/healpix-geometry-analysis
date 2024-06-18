import healpy as hp
import jax.numpy as jnp
import pytest
from healpix_geometry_analysis.coordinates import HealpixCoordinates
from numpy.testing import assert_allclose


@pytest.mark.parametrize("order", [0, 1, 2, 4, 8, 16, 20])
def test_equatorial_region(order):
    """Check if the equatorial region works as expected."""
    nside = 1 << order

    lon_ = jnp.radians(jnp.linspace(0.0, 90.0, 33))
    lat_ = jnp.arcsin(jnp.linspace(-2 / 3, 2 / 3 - 1 / nside, 33))
    lon, lat = jnp.meshgrid(lon_, lat_)

    # Use healpy to get the pixel centers
    tile_id = hp.ang2pix(nside, jnp.degrees(lon), jnp.degrees(lat), lonlat=True, nest=True)
    lonlat_center = hp.pix2ang(nside, tile_id, lonlat=True, nest=True)
    xyz_center = hp.ang2vec(*lonlat_center, lonlat=True)
    phi_center, z_center = jnp.radians(lonlat_center[0]), jnp.sin(jnp.radians(lonlat_center[1]))

    # Get the diagonal indices
    k_c = 3 * nside / 4 * (2 / 3 - z_center + 8 * phi_center / (3 * jnp.pi))
    kp_c = nside + 3 * nside / 4 * (2 / 3 - z_center - 8 * phi_center / (3 * jnp.pi))

    coords = HealpixCoordinates.from_order(order)
    phi, z = coords.phi_z(k_c, kp_c)

    assert_allclose(phi, phi_center, atol=1e-3 / nside, rtol=1e-3 / nside)
    assert_allclose(z, z_center, atol=1e-3 / nside, rtol=1e-3 / nside)

    lonlat_degrees = coords.lonlat_degrees(k_c, kp_c)
    assert_allclose(lonlat_degrees, lonlat_center, atol=1e-3 / nside, rtol=1e-3 / nside)

    xyz = coords.xyz(k_c, kp_c)
    assert_allclose(xyz, xyz_center.T, atol=1e-3 / nside, rtol=1e-3 / nside)


@pytest.mark.parametrize("order", [0, 1, 2, 5, 9, 13, 19])
def test_polar_region(order):
    """Check if the North polar region works good"""
    nside = 1 << order

    lon_ = jnp.radians(jnp.linspace(0.0, 90 - 1e-3 / nside, 33))
    lat_ = jnp.arcsin(jnp.linspace(2 / 3, 1, 33))
    lon, lat = jnp.meshgrid(lon_, lat_)

    # Use healpix to get the pixel centers
    tile_id = hp.ang2pix(nside, jnp.degrees(lon), jnp.degrees(lat), lonlat=True, nest=True)
    lonlat_center = hp.pix2ang(nside, tile_id, lonlat=True, nest=True)
    xyz_center = hp.ang2vec(*lonlat_center, lonlat=True)
    phi_center, z_center = jnp.radians(lonlat_center[0]), jnp.sin(jnp.radians(lonlat_center[1]))

    # Get the diagonal indices
    i_c = jnp.sqrt(3) * nside * jnp.sqrt(1 - z_center)
    j_c = 2 * i_c / jnp.pi * phi_center - 0.5
    k_c = j_c + 0.5
    kp_c = i_c - j_c - 0.5

    coords = HealpixCoordinates.from_order(order)
    phi, z = coords.phi_z(k_c, kp_c)

    assert_allclose(phi, phi_center, atol=1e-3 / nside, rtol=1e-3 / nside)
    assert_allclose(z, z_center, atol=1e-3 / nside, rtol=1e-3 / nside)

    lonlat_degrees = coords.lonlat_degrees(k_c, kp_c)
    assert_allclose(lonlat_degrees, lonlat_center, atol=1e-3 / nside, rtol=1e-3 / nside)

    xyz = coords.xyz(k_c, kp_c)
    assert_allclose(xyz, xyz_center.T, atol=1e-3 / nside, rtol=1e-3 / nside)
