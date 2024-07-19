"""Healpix tile coordinates math"""

import dataclasses
from typing import Self

import jax.numpy as jnp

from healpix_geometry_analysis.grid import HealpixGrid, HealpixGridPowerTwo


@dataclasses.dataclass
class HealpixCoordinates:
    """Healpix tile coordinates derived from diagonal indices

    Parameters
    ----------
    nside : int
        Healpix nside parameter, 2^order
    """

    grid: HealpixGrid
    """Healpix grid object specifying order"""

    @classmethod
    def from_order(cls, order: int) -> Self:
        """Create HealpixCoordinates using healpix order (depth)"""
        return cls(HealpixGridPowerTwo(order=order))

    @classmethod
    def from_nside(cls, nside: int) -> Self:
        """Create HealpixCoordinates using Nside parameter"""
        if 2 ** (order := int(jnp.log2(nside))) == nside:
            return cls.from_order(order)
        return cls(HealpixGrid(nside=nside))

    def xyz(self, k, kp):
        """Cartesian coordinates on the unit sphere from diagonal indices

        Parameters
        ----------
        k : float
            NW-SE diagonal index
        kp : float
            NE-SW diagonal index

        Returns
        -------
        x : float
            Cartesian x coordinate
        y : float
            Cartesian y coordinate
        z : float
            Cartesian z coordinate
        """
        phi, z = self.phi_z(k, kp)
        x = jnp.cos(phi) * jnp.sqrt(1 - z**2)
        y = jnp.sin(phi) * jnp.sqrt(1 - z**2)
        return x, y, z

    def lonlat_radians(self, k, kp):
        """Longitude and latitude in radians from diagonal indices

        Parameters
        ----------
        k : float
            SW-NE diagonal index
        kp : float
            SE-NW diagonal index

        Returns
        -------
        lon : float
            Longitude in radians
        lat : float
            Latitude in radians
        """
        phi, z = self.phi_z(k, kp)
        lon = phi
        lat = jnp.arcsin(z)
        return lon, lat

    def lonlat_degrees(self, k, kp):
        """Longitude and latitude in degrees from diagonal indices

        Parameters
        ----------
        k : float
            NW-SE diagonal index
        kp : float
            NE-SW diagonal index

        Returns
        -------
        lon : float
            Longitude in degrees
        lat : float
            Latitude in degrees
        """
        lon, lat = self.lonlat_radians(k, kp)
        return jnp.degrees(lon), jnp.degrees(lat)

    def phi_z(self, k, kp):
        """Cylindrical coordinates from diagonal indices

        Parameters
        ----------
        k : float
            NW-SE diagonal index
        kp : float
            NE-SW diagonal index

        Returns
        -------
        phi : float
            Longitude in radians
        z : float
            Sine of the latitude
        """
        eq_phi, eq_z = self._eq(k, kp)
        polar_phi, polar_z = self._polar(k, kp)
        z = jnp.where(eq_z <= 2 / 3, eq_z, polar_z)
        phi = jnp.where(eq_z <= 2 / 3, eq_phi, polar_phi)
        return phi, z

    def _eq(self, k, kp):
        """Cylidrical coordinates assuming the equatorial region"""
        z = 2 / 3 * (2 - (kp + k) / self.grid.nside)
        phi = jnp.pi / 4 / self.grid.nside * (self.grid.nside - kp + k)
        return phi, z

    def _polar(self, k, kp):
        """Cylindrical coordinates assuming the polar region"""
        j = jnp.abs(k) - 0.5
        i = jnp.abs(kp) + jnp.abs(k)

        z = 1 - (i / self.grid.nside) ** 2 / 3
        # Avoid division by zero when both k and kp are zero
        phi = jnp.where(i > 0, 0.5 * jnp.pi * (j + 0.5) / i, 0.0)
        phi = jnp.where(kp >= 0, phi, jnp.pi - phi)
        phi = jnp.where(k >= 0, phi, -phi)
        return phi, z

    def cos_arc(self, k1, kp1, k2, kp2):
        """Cosine of the great circle arc between two pixels"""
        phi1, z1 = self.phi_z(k1, kp1)
        phi2, z2 = self.phi_z(k2, kp2)
        return z1 * z2 + jnp.sqrt(1 - z1**2) * jnp.sqrt(1 - z2**2) * jnp.cos(phi1 - phi2)

    def chord_squared(self, k1, kp1, k2, kp2):
        """Square of chord distance between two pixels"""
        x1, y1, z1 = self.xyz(k1, kp1)
        x2, y2, z2 = self.xyz(k2, kp2)
        return jnp.square(x1 - x2) + jnp.square(y1 - y2) + jnp.square(z1 - z2)

    def diag_from_phi_z[T](self, phi: T, z: T) -> tuple[T, T]:
        """Diagonal indices from cylindrical coordinates"""
        k_eq, kp_eq = self._diag_eq(phi, z)
        k_pol, kp_pol = self._diag_pol(phi, z)

        k = jnp.where(z <= 2 / 3, k_eq, k_pol)
        kp = jnp.where(z <= 2 / 3, kp_eq, kp_pol)

        return k, kp

    def _diag_eq[T](self, phi: T, z: T) -> tuple[T, T]:
        """Diagonal indices assuming the equatorial region"""
        k = 3 * self.grid.nside / 4 * (2 / 3 - z + 8 * phi / (3 * jnp.pi))
        kp = self.grid.nside + 3 * self.grid.nside / 4 * (2 / 3 - z - 8 * phi / (3 * jnp.pi))
        return k, kp

    def _diag_pol[T](self, phi: T, z: T) -> tuple[T, T]:
        """Diagonal indices assuming the polar region"""
        i = jnp.sqrt(3) * self.grid.nside * jnp.sqrt(1 - z)
        j = 2 * i / jnp.pi * phi - 0.5
        k = j + 0.5
        kp = i - j - 0.5
        return k, kp

    def diag_from_lonlat_degrees[T](self, lon: T, lat: T) -> tuple[T, T]:
        """Diagonal indices from longitude and latitude in degrees

        Parameters
        ----------
        lon : float
            Longitude in degrees
        lat : float
            Latitude in degrees

        Returns
        -------
        k : float
            NW-SE diagonal index
        kp : float
            NE-SW diagonal index
        """
        phi, z = jnp.radians(lon), jnp.sin(jnp.radians(lat))
        return self.diag_from_phi_z(phi, z)
