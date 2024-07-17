import jax
import jax.numpy as jnp
import optax
from healpix_geometry_analysis.geometry.equatorial import EquatorialGeometry
from healpix_geometry_analysis.geometry.tile import TileGeometry
from healpix_geometry_analysis.problems.optax_optimizer import OptaxOptimizerProblem


def test_tile_problem_adabelief():
    """e2e test for TileProblem with Adam optimizer"""
    geometry = TileGeometry.from_order(
        order=0,
        k_center=0.5,
        kp_center=0.5,
        direction="p",
        distance="chord_squared",
    )
    problem = OptaxOptimizerProblem(geometry)

    optimizer = problem.freeze_optimizer(optax.adabelief(1e-1))
    rng_key = jax.random.PRNGKey(0)
    params = problem.initial_params(rng_key)
    opt_state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(problem.loss))

    for _ in range(100):
        loss, grads = value_and_grad(params)
        grads = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(
            params, problem.geometry.lower_bounds, problem.geometry.upper_bounds
        )

    arc_distance_deg = problem.geometry.arc_length_degrees(loss)
    assert arc_distance_deg < problem.geometry.coord.grid.average_pixel_size_degree


def test_meridian_problem_adamw():
    """e2e test for MeridianProblem with Adam optimizer with weight decay"""
    geometry = EquatorialGeometry.from_order(
        order=8,
        distance="chord_squared",
    )
    problem = OptaxOptimizerProblem(geometry)

    optimizer = problem.freeze_optimizer(optax.adamw(0.1))
    rng_key = jax.random.PRNGKey(0)
    params = problem.initial_params(rng_key)
    opt_state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(problem.loss))

    for _ in range(100):
        loss, grads = value_and_grad(params)
        grads = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(
            params, problem.geometry.lower_bounds, problem.geometry.upper_bounds
        )

    arc_distance_deg = problem.geometry.arc_length_degrees(loss)
    assert (
        arc_distance_deg < problem.geometry.coord.grid.average_pixel_size_degree
    ), f"min_distance parameters: {params}"
