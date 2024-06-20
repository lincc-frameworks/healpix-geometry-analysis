import dataclasses

from healpix_geometry_analysis.geometry.tile import TileGeometry


@dataclasses.dataclass
class BaseProblem:
    """Base class for the problems

    Parameters
    ----------
    geometry : TileGeometry
        Tile geometry object
    """

    geometry: TileGeometry
    """Geometry object describing k1, k2, kp1 & kp2 for the problem"""
