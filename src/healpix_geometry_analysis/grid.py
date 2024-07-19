"""Healpix grid utils"""

import dataclasses
import math
from functools import cached_property


@dataclasses.dataclass(frozen=True, kw_only=True)
class HealpixGrid:
    """Healpix grid properties"""

    nside: int
    """Number of tiles on each side of the base Healpix grid tile"""

    @cached_property
    def ntiles(self) -> int:
        """Number of tiles"""
        return 12 * self.nside**2

    @cached_property
    def tile_area_steradian(self) -> float:
        """Tile area in steradians"""
        return 4 * math.pi / self.ntiles

    @cached_property
    def average_pixel_size_radian(self) -> float:
        """Square root of the tile area in radians"""
        return math.sqrt(self.tile_area_steradian)

    @cached_property
    def average_pixel_size_degree(self) -> float:
        """Square root of the tile area in degrees"""
        return math.degrees(self.average_pixel_size_radian)

    @cached_property
    def average_pixel_size_arcmin(self) -> float:
        """Square root of the tile area in arcminutes"""
        return self.average_pixel_size_degree * 60

    @cached_property
    def average_pixel_size_arcsec(self) -> float:
        """Square root of the tile area in arcseconds"""
        return self.average_pixel_size_arcmin * 60


class HealpixGridPowerTwo(HealpixGrid):
    """Healpix grid with a notion of order, Nside = 2^order

    Parameters
    ----------
    order : int
        Healpix order (depth) of the grid
    """

    def __init__(self, *, order: int):
        super().__init__(nside=1 << order)
        self.order = order
