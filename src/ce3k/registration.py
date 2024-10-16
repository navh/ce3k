"""A JAX-version of OpenAI's infamous env.make(env_name)."""

from ce3k.environment import Environment


env_dict = {"dev-v0": Environment}
registered_environments = sorted([k for k in env_dict.keys()])


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's infamous env.make(env_name).


    Args:
      env_id: A string identifier for the environment.
      **env_kwargs: Keyword arguments to pass to the environment.


    Returns:
      A tuple of the environment and the default parameters.
    """

    if env_id in env_dict:
        env = env_dict[env_id](**env_kwargs)
    # if env_id == "dev-v0":
    # env = environment.ce3kEnv(**env_kwargs)
    else:
        raise ValueError(f"Environment ID '{env_id}' is not registered.")

    # Create a jax PRNG key for random seed control
    return env, env.default_params
