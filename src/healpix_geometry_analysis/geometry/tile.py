import dataclasses
from typing import Self

import jax.numpy as jnp

from healpix_geometry_analysis.coordinates import HealpixCoordinates
from healpix_geometry_analysis.geometry.base import DIRECTION_T, DIRECTIONS, DISTANCE_T, BaseGeometry


@dataclasses.dataclass(kw_only=True)
class TileGeometry(BaseGeometry):
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
    distance : {"chord_squared", "minus_cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "minus_cos_arc" for minus cosine of the great circle arc distance
    """

    k_center: float
    """NW-SE diagonal index of the pixel center"""

    kp_center: float
    """NE-SW diagonal index of the pixel center"""

    delta: float = 0.5
    """Offset in the diagonal index from the center to the pixel, typically 0.5"""

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
        distance : {"chord_squared", "minus_cos_arc"}
            Distance function to use:
            - "chord_squared" for squared chord distance in the unit sphere
            - "minus_cos_arc" for minus cosine of the great circle arc distance
        """
        coord = HealpixCoordinates.from_order(order)
        return cls(
            coord=coord, k_center=k_center, kp_center=kp_center, direction=direction, distance=distance
        )

    def diagonal_indices(self, params: dict[str, object]) -> tuple[object, object, object, object]:
        """Diagonal indices of the pixels

        Parameters
        ----------
        params : dict[str, object]
            Pytree with parameter values

        Returns
        -------
        tuple[object, object, object, object]
            Diagonal indices of the pixel: k1, k2, kp1, kp2
        """
        return params["k1"], params["k2"], params["kp1"], params["kp2"]

    parameter_names: tuple[str, str, str, str] = dataclasses.field(
        init=False, default=("k1", "k2", "kp1", "kp2")
    )

    @property
    def frozen_parameters(self) -> dict[str, object]:
        """Frozen parameters for the problem

        Returns
        -------
        tuple[str, str]
            Freezed parameters.
            ("k1" and "k2") for "m" direction
            and ("kp1" and "kp2") for "p" direction
        """
        if self.direction == "p":
            return {"kp1": self.kp_center - self.delta, "kp2": self.kp_center + self.delta}
        if self.direction == "m":
            return {"k1": self.k_center - self.delta, "k2": self.k_center + self.delta}
        raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")

    @property
    def free_parameter_limits(self) -> dict[str, tuple[object, object]]:
        """Free parameters for the problem and their limits

        Returns
        -------
        dict[str, tuple[object, object]]
            Free parameters and their lower and upper limits.
            ("k1" and "k2") for "m" direction
            and ("kp1" and "kp2") for "p" direction
        """
        if self.direction == "p":
            limits = (self.k_center - jnp.abs(self.delta), self.k_center + jnp.abs(self.delta))
            return dict.fromkeys(["k1", "k2"], limits)
        if self.direction == "m":
            limits = (self.kp_center - jnp.abs(self.delta), self.kp_center + jnp.abs(self.delta))
            return dict.fromkeys(["kp1", "kp2"], limits)
        raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
