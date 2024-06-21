"""Healpix tile coordinates math"""

import dataclasses
from typing import Self

import jax.numpy as jnp


@dataclasses.dataclass
class HealpixCoordinates:
    """Healpix tile coordinates derived from diagonal indices

    Parameters
    ----------
    nside : int
        Healpix nside parameter, 2^order
    """

    nside: int
    """Healpix nside parameter, 2^order"""

    @classmethod
    def from_order(cls, order: int) -> Self:
        """Create HealpixCoordinates using healpix order (depth)"""
        return cls(nside=1 << order)

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
        z = 2 / 3 * (2 - (kp + k) / self.nside)
        phi = jnp.pi / 4 / self.nside * (self.nside - kp + k)
        return phi, z

    def _polar(self, k, kp):
        """Cylindrical coordinates assuming the polar region"""
        j = jnp.abs(k) - 0.5
        i = jnp.abs(kp) + jnp.abs(k)

        z = 1 - (i / self.nside) ** 2 / 3
        phi = 0.5 * jnp.pi * (j + 0.5) / i
        phi = jnp.where(kp >= 0, phi, jnp.pi - phi)
        phi = jnp.where(k >= 0, phi, -phi)
        return phi, z
