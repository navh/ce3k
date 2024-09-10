from ce3k.situations import opensea
from ce3k.targets import plane
from ce3k.platforms import Rock
from ce3k.sensors import s_band_radar
from ce3k.functions import search, single_target_tracker

rock = Rock(sensors=[s_band_radar], functions=[search, single_target_tracker])


make_env = env_creator(
    situation=opensea,
    targets=[plane],
    platform=rock,
)
env = make_env()
