import dataclasses
from typing import Self

import jax.numpy as jnp

from healpix_geometry_analysis.coordinates import HealpixCoordinates
from healpix_geometry_analysis.geometry.base import (
    DIRECTION_T,
    DIRECTIONS,
    DISTANCE_T,
    REGION,
    REGION_T,
    BaseGeometry,
)


@dataclasses.dataclass(kw_only=True)
class MeridianGeometry(BaseGeometry):
    """Distance problem for two opposite edges of a Healpix tile

    Parameters
    ----------
    coord : HealpixCoordinates
        Healpix coordinates object
    direction : {"p", "m"}
        direction of edges of the tile to compare:
        - "p" (plus) for NE and SW edges
        - "m" (minus) for NW and SE edges
    distance : {"chord_squared", "minus_cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "minus_cos_arc" for minus cosine of the great circle arc distance
    region : {"equator", "polar"}
        Region of the floating tile:
        - "equator" for equatorial region, z_center < 2/3
        - "polar" for polar region, z_center >= 2/3
    delta : float, optional
        Offset in the diagonal index from the center to the pixel, default is 0.5
    """

    region: REGION_T

    delta: float = 0.5
    """Offset in the diagonal index from the center to the pixel, typically 0.5"""

    def __post_init__(self):
        super().__post_init__()

        if self.region == "equator" and self.direction == "m":
            raise ValueError(
                "Equatorial region is symmetric over direction, use direction 'p' instead of 'm'"
            )

        if self.region == "equator":
            # From the equator up to the corner of the equatorial face
            self.z_center_limits = 0.0, 2 / 3 * (1 - (2 * self.delta) / self.coord.grid.nside)
        elif self.region == "polar":
            # From the corner of the polar face to the pole tile
            self.z_center_limits = 2 / 3, 1 - (2 * self.delta) ** 2 / (3 * self.coord.grid.nside**2)
        else:
            raise ValueError(f"Invalid region: {self.region}, must be one of {REGION}")

    @classmethod
    def from_order(
        cls,
        order: int,
        *,
        direction: DIRECTION_T,
        distance: DISTANCE_T,
        region: REGION_T,
    ) -> Self:
        """Create TileProblem using order and diagonal indices

        Parameters
        ----------
        order : int
            Healpix order (depth) of the coord
        direction : {"p", "m"}
            direction of edges of the tile to compare:
            - "p" (plus) for NE and SW edges
            - "m" (minus) for NW and SE edges
        distance : {"chord_squared", "minus_cos_arc"}
            Distance function to use:
            - "chord_squared" for squared chord distance in the unit sphere
            - "minus_cos_arc" for minus cosine of the great circle arc distance
        region : {"equator", "polar"}
            Region of the floating tile:
            - "equator" for equatorial region, z_center < 2/3
            - "polar" for polar region, z_center >= 2/3
        """
        coord = HealpixCoordinates.from_order(order)
        return cls(
            coord=coord,
            direction=direction,
            distance=distance,
            region=region,
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

        if self.region == "equator":
            # For equatorial region we may use any phi
            phi_center = 0.0
        elif self.region == "polar":
            # For polar region we use phi corresponding to j=delta=0.5
            i_from_z = self.coord.i_from_z(params["z_center"])
            phi_center = self.delta**2 * jnp.pi / i_from_z
        else:
            raise ValueError(f"Invalid region: {self.region}, must be one of {REGION}")

        k_center, kp_center = self.coord.diag_from_phi_z(phi_center, params["z_center"])
        return (
            k_center + params["delta_k1"],
            k_center + params["delta_k2"],
            kp_center + params["delta_kp1"],
            kp_center + params["delta_kp2"],
        )

    parameter_names: tuple[str, str, str, str, str] = dataclasses.field(
        init=False, default=("z_center", "delta_k1", "delta_k2", "delta_kp1", "delta_kp2")
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
            return {"delta_kp1": -self.delta, "delta_kp2": self.delta}
        if self.direction == "m":
            return {"delta_k1": -self.delta, "delta_k2": self.delta}
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
            diag_names = "delta_k1", "delta_k2"
        elif self.direction == "m":
            diag_names = "delta_kp1", "delta_kp2"
        else:
            raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
        diag_limits = -jnp.abs(self.delta), jnp.abs(self.delta)

        return {"z_center": self.z_center_limits} | dict.fromkeys(diag_names, diag_limits)
