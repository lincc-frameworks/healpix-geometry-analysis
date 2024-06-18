import dataclasses
from typing import Literal, Self

import numpyro
import numpyro.distributions as dist

from healpix_geometry_analysis.coordinates import HealpixCoordinates

DIRECTIONS = ["p", "m"]
DIRECTION_T = Literal[*DIRECTIONS]

DISTANCE = ["chord_squared", "cos_arc"]
DISTANCE_T = Literal[*DISTANCE]


@dataclasses.dataclass(kw_only=True)
class TileProblem:
    """Distance problem for two opposite edges of a Healpix tile

    Parameters
    ----------
    coord : HealpixCoordinates
        Healpix coordinates object
    k_center : float
        NW-SE diagonal index of the pixel center
    kp_center : float
        NE-SW diagonal index of the pixel center
    direction : {"p", "m"}
        direction of edges of the tile to compare:
        - "p" (plus) for NE and SW edges
        - "m" (minus) for NW and SE edges
    distance : {"chord_squared", "cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "cos_arc" for cosine of the great circle arc distance
    """

    delta: float = 0.5
    """Offset in the diagonal index from the center to the pixel, typically 0.5"""

    coord: HealpixCoordinates
    """Healpix coord object specifying order"""

    k_center: float
    """NW-SE diagonal indexx"""

    kp_center: float
    """NE-SW diagonal index"""

    direction: DIRECTION_T
    """direction of edges of the tile to compare, "p" (plus) or "m" (minus)"""

    distance: DISTANCE_T
    """Distance function to use, "chord_squared" or "cos_arc\""""

    def __post_init__(self):
        assert (
            self.direction in DIRECTIONS
        ), f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}"

    @classmethod
    def from_order(
        cls, order: int, *, k_center: float, kp_center: float, direction: DIRECTION_T, distance: DISTANCE_T
    ) -> Self:
        """Create TileProblem using order and diagonal indices

        Parameters
        ----------
        order : int
            Healpix order (depth) of the coord
        k_center : float
            NW-SE diagonal index of the pixel center
        kp_center : float
            NE-SW diagonal index of the pixel center
        direction : {"p", "m"}
            direction of edges of the tile to compare:
            - "p" (plus) for NE and SW edges
            - "m" (minus) for NW and SE edges
        distance : {"chord_squared", "cos_arc"}
            Distance function to use:
            - "chord_squared" for squared chord distance in the unit sphere
            - "cos_arc" for cosine of the great circle arc distance

        Returns
        -------
        TileProblem
            TileProblem object
        """
        return cls(
            coord=HealpixCoordinates.from_order(order),
            k_center=k_center,
            kp_center=kp_center,
            direction=direction,
            distance=distance,
        )

    def side1(self):
        """Get k and kp indices for the first side of the tile

        It is NE for "p" direction and NW for "m" direction
        """
        if self.direction == "p":
            k = numpyro.sample("k1", dist.Uniform(self.k_center - self.delta, self.k_center + self.delta))
            kp = self.kp_center - self.delta
        elif self.direction == "m":
            k = self.k_center - self.delta
            kp = numpyro.sample("kp1", dist.Uniform(self.kp_center - self.delta, self.kp_center + self.delta))
        else:
            raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
        return k, kp

    def side2(self):
        """Get k and kp indices for the second side of the tile

        It is SW for "p" direction and SE for "m" direction
        """
        if self.direction == "p":
            k = numpyro.sample("k2", dist.Uniform(self.k_center - self.delta, self.k_center + self.delta))
            kp = self.kp_center + self.delta
        elif self.direction == "m":
            k = self.k_center + self.delta
            kp = numpyro.sample("kp2", dist.Uniform(self.kp_center - self.delta, self.kp_center + self.delta))
        else:
            raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
        return k, kp

    def model(self):
        """Numpyro model for the tile distance problem"""
        k1, kp1 = self.side1()
        k2, kp2 = self.side2()

        if self.distance == "chord_squared":
            distance = self.coord.chord_squared(k1, kp1, k2, kp2)
            # Use negative distance to minimize it
            numpyro.factor("target", -distance)
        elif self.distance == "cos_arc":
            distance = self.coord.cos_arc(k1, kp1, k2, kp2)
            # Use positive cosine to maximize it and minimize distance
            numpyro.factor("target", distance)
        else:
            raise ValueError(f"Invalid distance: {self.distance}, must be one of {DISTANCE}")

        numpyro.deterministic("distance", distance)
