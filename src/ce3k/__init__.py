# def hello() -> None:
# print("Hello from ce3k!")

from gymnasium.envs.registration import register

register(
    id="navh/ce3kEnv-v0",
    entry_point="ce3k.environment:ce3kEnv",
)
