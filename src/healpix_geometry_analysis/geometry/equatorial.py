import dataclasses
from typing import Self

from healpix_geometry_analysis.coordinates import HealpixCoordinates
from healpix_geometry_analysis.geometry.base import (
    DIRECTION_T,
    DISTANCE_T,
    BaseGeometry,
)


@dataclasses.dataclass(kw_only=True)
class EquatorialGeometry(BaseGeometry):
    """Problem for two opposite edges of a floating tile in the equatorial region.

    It finds a minimum distance over two parallel diagonal lines in the equatorial region.

    It is symmetric over the direction, so only one direction is needed.

    Parameters
    ----------
    coord : HealpixCoordinates
        Healpix coordinates object
    distance : {"chord_squared", "minus_cos_arc"}
        Distance function to use:
        - "chord_squared" for squared chord distance in the unit sphere
        - "minus_cos_arc" for minus cosine of the great circle arc distance
    delta : float, optional
        Offset in the diagonal index from the center to the pixel, default is 0.5
    """

    direction: DIRECTION_T = dataclasses.field(default="m", init=False)
    """Direction of edges of the tile to compare, "m" (minus) for NW and SE edges"""

    delta: float = 0.5
    """Offset in the diagonal index from the center to the pixel, typically 0.5"""

    def __post_init__(self):
        super().__post_init__()

    @classmethod
    def from_order(
        cls,
        order: int,
        *,
        distance: DISTANCE_T,
    ) -> Self:
        """Create TileProblem using order and diagonal indices

        Parameters
        ----------
        order : int
            Healpix order (depth) of the coord
        distance : {"chord_squared", "minus_cos_arc"}
            Distance function to use:
            - "chord_squared" for squared chord distance in the unit sphere
            - "minus_cos_arc" for minus cosine of the great circle arc distance
        """
        coord = HealpixCoordinates.from_order(order)
        return cls(
            coord=coord,
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

        return (
            params["k1"],
            params["k2"],
            params["kp1"],
            params["kp2_minus_kp1"] + params["kp1"],
        )

    parameter_names: tuple[str, str, str, str, str] = dataclasses.field(
        init=False, default=("k1", "k2", "kp1", "kp2_minus_kp1")
    )

    @property
    def frozen_parameters(self) -> dict[str, object]:
        """Frozen parameters for the problem

        Returns
        -------
        tuple[str, str]
            Frozen parameters.
            ("k1" and "k2")
        """
        if self.direction != "m":
            raise ValueError(f'Invalid direction: {self.direction}, must be "m"')
        k1 = 0.5 * self.coord.grid.nside
        return {"k1": k1, "k2": k1 + 2.0 * self.delta}

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
        if self.direction != "m":
            raise ValueError(f'Invalid direction: {self.direction}, must be one of "m"')
        return {
            "kp1": (0.5 * self.coord.grid.nside, self.coord.grid.nside * 1.5),
            "kp2_minus_kp1": (-2.0 * self.delta, 2.0 * self.delta),
        }
