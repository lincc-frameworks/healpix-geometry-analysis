import dataclasses
from typing import Self

import jax.numpy as jnp

from healpix_geometry_analysis.coordinates import HealpixCoordinates
from healpix_geometry_analysis.geometry.base import (
    DIRECTION_T,
    DIRECTIONS,
    DISTANCE_T,
    BaseGeometry,
)


@dataclasses.dataclass(kw_only=True)
class IntermediateGeometry(BaseGeometry):
    """Problem for two opposite edges of a floating tile between the regions

    Parameters
    ----------
    coord : HealpixCoordinates
        Healpix coordinates object
    direction : {"p", "m"}
        direction of edges of the tile to compare:
        - "p" (plus) for NW and SE edges
        - "m" (minus) for NE and SW edges
    distance : {"chord_squared", "minus_cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "minus_cos_arc" for minus cosine of the great circle arc distance
    delta : float, optional
        Offset in the diagonal index from the center to the pixel, default is 0.5
    """

    delta: float = 0.5
    """Offset in the diagonal index from the center to the pixel, typically 0.5"""

    def __post_init__(self):
        super().__post_init__()

        self.z_center = 2 / 3

        min_delta_phi_from_meridian = 0.5 * self.delta * jnp.pi / self.coord.grid.nside
        self.phi_center_limits = min_delta_phi_from_meridian, 0.5 * jnp.pi - min_delta_phi_from_meridian

    @classmethod
    def from_order(
        cls,
        order: int,
        *,
        direction: DIRECTION_T,
        distance: DISTANCE_T,
    ) -> Self:
        """Create TileProblem using order and diagonal indices

        Parameters
        ----------
        order : int
            Healpix order (depth) of the coord
        direction : {"p", "m"}
            direction of edges of the tile to compare:
            - "p" (plus) for NW and SS edges
            - "m" (minus) for NE and SW edges
        distance : {"chord_squared", "minus_cos_arc"}
            Distance function to use:
            - "chord_squared" for squared chord distance in the unit sphere
            - "minus_cos_arc" for minus cosine of the great circle arc distance
        """
        coord = HealpixCoordinates.from_order(order)
        return cls(
            coord=coord,
            direction=direction,
            distance=distance,
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
        k_center, kp_center = self.coord.diag_from_phi_z(params["phi_center"], self.z_center)
        return (
            k_center + params["delta_k1"],
            k_center + params["delta_k2"],
            kp_center + params["delta_kp1"],
            kp_center + params["delta_kp2"],
        )

    parameter_names: tuple[str, str, str, str, str] = dataclasses.field(
        init=False, default=("phi_center", "delta_k1", "delta_k2", "delta_kp1", "delta_kp2")
    )

    @property
    def frozen_parameters(self) -> dict[str, object]:
        """Frozen parameters for the problem

        Returns
        -------
        tuple[str, str]
            Frozen parameters.
            ("k1" and "k2") for "p" direction
            and ("kp1" and "kp2") for "m" direction
        """
        if self.direction == "p":
            return {"delta_k1": -self.delta, "delta_k2": self.delta}
        if self.direction == "m":
            return {"delta_kp1": -self.delta, "delta_kp2": self.delta}
        raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")

    @property
    def free_parameter_limits(self) -> dict[str, tuple[object, object]]:
        """Free parameters for the problem and their limits

        Returns
        -------
        dict[str, tuple[object, object]]
            Free parameters and their lower and upper limits.
            ("kp1" and "kp2") for "p" direction
            and ("k1" and "k2") for "m" direction
        """
        if self.direction == "p":
            diag_names = "delta_kp1", "delta_kp2"
        elif self.direction == "m":
            diag_names = "delta_k1", "delta_k2"
        else:
            raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
        diag_limits = -jnp.abs(self.delta), jnp.abs(self.delta)

        return {"phi_center": self.phi_center_limits} | dict.fromkeys(diag_names, diag_limits)
