import jax


def enable_x64():
    """Make Jax to use double precision by default

    It must be run before any other Jax code.
    """
    jax.config.update("jax_enable_x64", True)
