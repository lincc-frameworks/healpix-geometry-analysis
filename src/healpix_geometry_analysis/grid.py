"""Healpix grid utils"""

import dataclasses
import math
from functools import cached_property


@dataclasses.dataclass(frozen=True)
class HealpixGrid:
    """Healpix grid properties"""

    order: int
    """Healpix order (depth) of the grid"""

    @cached_property
    def nside(self) -> int:
        """Number of tiles on each side of the base Healpix grid tile"""
        return 1 << self.order

    @cached_property
    def ntiles(self) -> int:
        """Number of tiles"""
        return 12 * (1 << (2 * self.order))

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
