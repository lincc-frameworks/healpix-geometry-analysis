import dataclasses

import jax.numpy as jnp

from healpix_geometry_analysis.geometry.base import (
    DIRECTIONS,
    BaseGeometry,
)


@dataclasses.dataclass(kw_only=True)
class PolarGeometry(BaseGeometry):
    """Problem for two opposite edges of a floating tile in the polar region.

    The position of the center of the tile is parameterized with
    vertical index i and relative diagonal coordinate k_to_k_range
    (we use it as a horizontal coordinate). This relative
    index takes values from 0 to 1, where 0 corresponds to the leftmost
    tile, k=0.5 (adjacent to the zero meridian), and 1 to the rightmost tile,
    k=0.5*i, which corresponds to phi = pi/4, and located in the center
    of the "face" (order 0) tile.

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

        self.i_center_limits = (2.0 * self.delta, self.coord.grid.nside - 2.0 * self.delta)
        self.k_to_k_range_center_limits = (0.0, 1.0)

    def diagonal_indices[T](self, params: dict[str, T]) -> tuple[T, T, T, T]:
        """Diagonal indices of the pixels

        Parameters
        ----------
        params : dict[str, T]
            Pytree with parameter values

        Returns
        -------
        tuple[T, T, T, T]
            Diagonal indices of the pixel: k1, k2, kp1, kp2
        """
        i_center = params["i_center"]
        k_range_length = 0.5 * i_center - self.delta
        k_center = params["k_to_k_range_center"] * k_range_length + self.delta
        kp_center = self.coord.kp_from_k_i(k_center, i_center)

        return (
            k_center + params["delta_k1"],
            k_center + params["delta_k2"],
            kp_center + params["delta_kp1"],
            kp_center + params["delta_kp2"],
        )

    parameter_names: tuple[str, str, str, str, str] = dataclasses.field(
        init=False,
        default=("i_center", "k_to_k_range_center", "delta_k1", "delta_k2", "delta_kp1", "delta_kp2"),
    )

    @property
    def frozen_parameters(self) -> dict[str, object]:
        """Frozen parameters for the problem

        Returns
        -------
        tuple[str, str]
            Frozen parameters.
            ("delta_k1" and "delta_k2") for "p" direction
            and ("delta_kp1" and "delta_kp2") for "m" direction
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
            "i_center", "k_to_k_range_center" and ("kp1" and "kp2") for "p" direction
            or ("k1" and "k2") for "m" direction
        """
        if self.direction == "p":
            diag_names = "delta_kp1", "delta_kp2"
        elif self.direction == "m":
            diag_names = "delta_k1", "delta_k2"
        else:
            raise ValueError(f"Invalid direction: {self.direction}, must be one of {DIRECTIONS}")
        diag_limits = -jnp.abs(self.delta), jnp.abs(self.delta)

        return {
            "i_center": self.i_center_limits,
            "k_to_k_range_center": self.k_to_k_range_center_limits,
        } | dict.fromkeys(diag_names, diag_limits)
