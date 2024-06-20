import jax
import optax
from healpix_geometry_analysis.geometry.tile import TileGeometry
from healpix_geometry_analysis.problems.optax_optimizer import OptaxOptimizerProblem


def test_tile_problem_adabelief():
    """ "e2e test for TileProblem with Adam optimizer"""
    geometry = TileGeometry.from_order(
        order=4,
        k_center=1.5,
        kp_center=14.5,
        direction="p",
        distance="chord_squared",
    )
    problem = OptaxOptimizerProblem(geometry)

    optimizer = problem.freeze_optimizer(optax.adabelief(1e-1))
    rng_key = jax.random.PRNGKey(0)
    params = problem.initial_params(rng_key)
    opt_state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(problem.optix_loss))

    for _ in range(100):
        loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(
            params, problem.geometry.lower_bounds, problem.geometry.upper_bounds
        )

    arc_distance_deg = problem.geometry.arc_length_degrees(loss)
    assert arc_distance_deg < problem.geometry.coord.grid.average_pixel_size_degree
