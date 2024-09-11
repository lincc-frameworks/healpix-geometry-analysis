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
    actual_k_c, actual_kp_c = coords.diag_from_lonlat_degrees(*lonlat_center)
    assert_allclose(k_c, actual_k_c)
    assert_allclose(kp_c, actual_kp_c)

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
    actual_k_c, actual_kp_c = coords.diag_from_lonlat_degrees(*lonlat_center)
    assert_allclose(k_c, actual_k_c)
    assert_allclose(kp_c, actual_kp_c)

    phi, z = coords.phi_z(k_c, kp_c)

    assert_allclose(phi, phi_center, atol=1e-3 / nside, rtol=1e-3 / nside)
    assert_allclose(z, z_center, atol=1e-3 / nside, rtol=1e-3 / nside)

    lonlat_degrees = coords.lonlat_degrees(k_c, kp_c)
    assert_allclose(lonlat_degrees, lonlat_center, atol=1e-3 / nside, rtol=1e-3 / nside)

    xyz = coords.xyz(k_c, kp_c)
    assert_allclose(xyz, xyz_center.T, atol=1e-3 / nside, rtol=1e-3 / nside)


@pytest.mark.parametrize("nside", [1, 2, 3, 4, 8, 16, 17, 32, 128])
def test_unique_equatorial_tiles(nside):
    """Check if the unique equatorial tiles are computed correctly"""
    coords = HealpixCoordinates.from_nside(nside)
    k, kp = coords.unique_equatorial_tiles()

    assert k.shape == kp.shape
    assert k.shape == (nside,)

    phi, z = coords.phi_z(k, kp)
    assert jnp.sum(jnp.abs(phi) < 1e-8) == (nside + 1) // 2
    assert jnp.sum(jnp.abs(phi) > 1e-8) == nside // 2
    assert jnp.all(z < 2 / 3)
    assert jnp.all(z >= 0)


@pytest.mark.parametrize("nside", [1, 2, 3, 4, 7, 8, 16, 32, 77, 128])
def test_unique_intermediate_tiles(nside):
    """Check if the unique intermediate tiles are computed correctly"""
    coords = HealpixCoordinates.from_nside(nside)
    k, kp = coords.unique_intermediate_tiles()

    assert k.shape == kp.shape
    assert k.shape == ((nside + 1) // 2,)

    phi, z = coords.phi_z(k, kp)
    assert jnp.allclose(z, 2 / 3)
    assert jnp.all(phi >= 0)
    assert jnp.sum(phi > jnp.pi / 4 - 1e-6) == nside % 2


@pytest.mark.parametrize("nside", [1, 2, 3, 7])
def test_unique_polar_tiles(nside):
    """Check if the unique polar tiles are computed correctly"""
    coords = HealpixCoordinates.from_nside(nside)
    k, kp = coords.unique_polar_tiles()

    assert k.shape == kp.shape
    assert k.size == (nside + 1) // 2 * (nside // 2)

    phi, z = coords.phi_z(k, kp)
    assert jnp.all(z > 2 / 3)
    assert jnp.all(z < 1)
    assert jnp.all(phi > 0)
    assert jnp.sum(phi > jnp.pi / 4 - 1e-6) == nside // 2


@pytest.mark.parametrize("nside", [1, 2, 7, 8])
def test_unique_tiles(nside):
    """Check if the unique tiles are computed correctly"""
    coords = HealpixCoordinates.from_nside(nside)
    k, kp = coords.unique_tiles()

    assert k.shape == kp.shape
    assert k.size == nside + (nside + 1) // 2 + (nside + 1) // 2 * (nside // 2)

    phi, z = coords.phi_z(k, kp)
    assert jnp.all(z >= 0)
    assert jnp.all(z < 1)
    assert jnp.all(phi >= -1e-8)
    assert jnp.all(phi <= jnp.pi / 4 + 1e-8)
