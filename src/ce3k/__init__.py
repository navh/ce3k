# def hello() -> None:
# print("Hello from ce3k!")

# from gymnasium.envs.registration import register

# register(
# id="navh/ce3kEnv-v0",
# entry_point="ce3k.environment:ce3kEnv",
# )


"""Gymnax: A library for creating and registering Gym environments."""

# from ce3k import environment
from ce3k import registration

# EnvParams = environments.EnvParams
# EnvState = environments.EnvState
make = registration.make
registered_envs = registration.registered_environments


# __all__ = ["make", "registered_envs", "EnvState", "EnvParams"]
__all__ = ["make", "registered_envs"]
