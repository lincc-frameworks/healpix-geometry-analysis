import jax
import jax.numpy as jnp
import optax
from healpix_geometry_analysis.problems.tile import TileProblem
from numpyro.infer import MCMC, NUTS


def test_tile_problem_nuts():
    """e2e test for TileProblem with MCMC sampler"""
    problem = TileProblem.from_order(
        order=4,
        k_center=1.5,
        kp_center=14.5,
        direction="p",
        distance="chord_squared",
    )

    kernel = NUTS(problem.numpyro_model)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=100)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    arc_distance_deg = jnp.degrees(2.0 * jnp.arcsin(0.5 * jnp.sqrt(samples["distance"])))

    argmin = jnp.argmin(arc_distance_deg)
    min_distance = arc_distance_deg[argmin]
    assert min_distance < problem.coord.grid.average_pixel_size_degree


def test_tile_problem_adam():
    """ "e2e test for TileProblem with Adam optimizer"""
    problem = TileProblem.from_order(
        order=7,
        k_center=(1 << 7) - 10.5,
        kp_center=7.5,
        direction="m",
        distance="chord_squared",
    )

    optimizer = problem.freeze_optax_optimizer(optax.adabelief(1e-1))
    rng_key = jax.random.PRNGKey(0)
    params = problem.params(rng_key)
    opt_state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(problem.optix_loss))

    for _ in range(100):
        loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(params, problem.lower_bounds, problem.upper_bounds)

    arc_distance_deg = jnp.degrees(2.0 * jnp.arcsin(0.5 * jnp.sqrt(loss)))
    assert arc_distance_deg < problem.coord.grid.average_pixel_size_degree
