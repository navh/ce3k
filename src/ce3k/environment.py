import jax.numpy as jnp
from jax.scipy.linalg import block_diag
import jax
from typing import TypeVar
from typing import Generic
from typing import Any
from gymnax.environments import spaces

from flax import struct

TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")

# UPDATE: so I'm just using `jnp.finfo(DTYPE_FLOAT).max`, this is present in a bunch of gymnax envs.
# Okay so initially I was thinking about doing time in milliseconds and uint16 gives me 60 seconds.
# For now, this limits how long into the future you can request a new look, and nothing else.
# I'm going to leave it at 65.44 seconds, a suspicious number, so that hopefully if it comes up this will scream at me
# TMAX: float = 65.44
# COSTMAX: float = 65.44

# DTYPE_FLOAT = jnp.float16  # f16 can't represent 200_000m, I've moved everything to km so flip this back on and see if it's any faster.
# DTYPE_INT = jnp.int16
DTYPE_FLOAT = jnp.float32
DTYPE_INT = jnp.int32

MAX_TARGETS = 2**6
AZ_SLICES = 15
EL_SLICES = 5
SENSORS = 1

SENSOR_MIN_AZ = 0
SENSOR_MAX_AZ = jnp.pi / 2  # 90 degrees
SENSOR_MIN_EL = 0
# 90/15 = 6 degrees per slice, 6*5 = 30 degrees total
SENSOR_MAX_EL = jnp.pi / 6  # 30 degrees


# Initially this was divided up into a "theatre" and "platform" spaces, for now I'm prepending "ship" to the platform bits
@struct.dataclass
class EnvState:
    target_positions: jax.Array
    target_singer_sigmas: jax.Array
    target_singer_thetas: jax.Array

    # platform has
    ship_state: jax.Array  # should we get this puppy moving? for now, origin
    ship_idle_sensor: int  # which sensor is free?
    ship_sensor_cooldowns: jax.Array  # time until each sensor is free
    ship_searchers: jax.Array  # so this is the wedge countdown?
    tracker_t_desireds: jax.Array
    tracker_t_deadlines: jax.Array
    tracker_t_dwell_estimates: jax.Array


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 1
    new_target_rate_parameter: float = 0.05  # lambda in the poisson distribution
    target_max_v: float = 1000  # m/s
    target_max_a: float = 0  # m/s^2 # zero, should be N(0,Sigma)?

    # Range was 10_000m to 184_000m in the slides.
    arena_min_dx: float = 0  # m
    arena_max_dx: float = 184_000  # m
    arena_min_dy: float = 0  # m
    arena_max_dy: float = 184_000  # m
    arena_min_dz: float = 0  # m
    # arena_min_dz: float = 10_000  # m
    arena_max_dz: float = 20_000  # m
    # More thoughts, uint16 is 65535, so 200_000/65535 is 3.05m
    # On one hand, 3m is pretty good resolution.
    # On the other hand, moving at speeds of 1000m/s, 3m is 3ms of travel.
    # Many of my dwells are in the ms range, and many of the speeds are in the 100s of m/s.
    # I don't think I can get away with 3m resolution.
    # 200_000/(2**32) is 0.046m, which is probably overkill.
    # 20_000/2**16 is 0.3m, which is probably good enough?
    # Eh, I'll go with f32 for now, but I'd really like to try to cram this into 16 somehow.
    # It may be more interesting to do a slightly smaller arena but with faster and more targets.
    # Simulating this gigantic void with mm resolution feels wasteful.

    search_zero_cost_revisit_interval: float = 1.5
    search_dwell_duration: float = 0.01  # from Sunila's slides
    search_penalty_per_second_late: int = 1
    track_penalty_per_second_late: int = 1


class Environment(Generic[TEnvState, TEnvParams]):  # object):
    """Jittable abstract base class for all gymnax Environments."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def execute_search(
        self, state: EnvState, action: int, params: EnvParams
    ) -> EnvState:
        # Find the least recently used beam and execute it
        # check the search pattern
        # transform any of the targets found during the search into 'trackers'

        # For each target, calculate the SNR based on the action
        # SNR is defined as zero if the target is not in the beam selected by the action
        #
        # azimuth_bottom = (action % AZ_SLICES) * (
        #     SENSOR_MAX_AZ - SENSOR_MIN_AZ
        # ) / AZ_SLICES - SENSOR_MIN_AZ
        # azimuth_top = (action % AZ_SLICES + 1) * (
        #     SENSOR_MAX_AZ - SENSOR_MIN_AZ
        # ) / AZ_SLICES - SENSOR_MIN_AZ
        #
        # elevation_bottom = (action // AZ_SLICES) * (
        #     SENSOR_MAX_EL - SENSOR_MIN_EL
        # ) / EL_SLICES - SENSOR_MIN_EL
        # elevation_top = (action // AZ_SLICES + 1) * (
        #     SENSOR_MAX_EL - SENSOR_MIN_EL
        # ) / EL_SLICES - SENSOR_MIN_EL
        #
        # target_sphere_from_origin = jnp.array(
        #     [
        #         jnp.sqrt(
        #             jnp.array(
        #                 [
        #                     state.target_positions[0] ** 2,
        #                     state.target_positions[3] ** 2,
        #                     state.target_positions[6] ** 2,
        #                 ]
        #             )
        #         ),  # range r
        #         jnp.arccos(
        #             state.target_positions[6]
        #             / jnp.linalg.norm(
        #                 jnp.array(
        #                     [
        #                         state.target_positions[0],
        #                         state.target_positions[3],
        #                         state.target_positions[6],
        #                     ]
        #                 )
        #             ),
        #         ),  # elevation theta
        #         jnp.arctan2(
        #             state.target_positions[3], state.target_positions[0]
        #         ),  # azimuth phi
        #     ]
        # )
        #
        # target_in_beam = (
        #     (targets_sphere_from_origin[1] >= elevation_bottom)
        #     & (targets_sphere_from_origin[1] < elevation_top)
        #     & (targets_sphere_from_origin[2] >= azimuth_bottom)
        #     & (targets_sphere_from_origin[2] < azimuth_top)
        # )
        #
        return EnvState(
            state.target_positions,
            state.target_singer_sigmas,
            state.target_singer_thetas,
            state.ship_state,
            state.ship_idle_sensor,
            state.ship_sensor_cooldowns.at[state.ship_idle_sensor].set(
                params.search_dwell_duration
            ),
            state.ship_searchers.at[state.ship_searchers.argmin()].set(
                params.search_zero_cost_revisit_interval
            ),  # least recently used sensor gets bumped up to revisit interval
            state.tracker_t_desireds,
            state.tracker_t_deadlines,
            state.tracker_t_dwell_estimates,
        )

    def execute_track(
        self, state: EnvState, action: int, params: EnvParams
    ) -> EnvState:
        # Execute the tracker indicitaed by the action number.
        #
        # TODO: Somewhere here we need a mechanism to stop a tracker.
        track_index = action - SENSORS

        # Calculate actual dwell time
        t_c_super_n = 0.01  # 10ms from the slides
        r_0 = 184_000  # 184km from the slides
        r = jnp.sqrt(
            state.target_positions[track_index][0] ** 2
            + state.target_positions[track_index][3] ** 2
            + state.target_positions[track_index][6] ** 2
        )
        # wait, I already need to know the range to calculate the dwell time?
        # I guess this should be coming from a kalman filter or something.
        SN_0 = 40  # 16dB from the slides?
        SNR = 40  # TODO: This is wrong, @Sunila
        new_t_dwell_estimate = t_c_super_n * (r / r_0) ** 4  # Ignore SN_0?
        # @Sunila - something to check

        # Calculate new time desired
        # @Sunila todo fix sigma_theta as well
        sigma_theta = 1  # There's a sigma theta here, I'm worried it's some 2d stuff.
        u = 0.3  # Magic number from the slides.
        new_t_desired = (
            0.4
            * (
                r
                * sigma_theta
                * jnp.sqrt(state.target_singer_thetas[track_index][0])
                / state.target_singer_sigmas[track_index][0]
            )  # TODO: this '[0]' doesn't feel necessary, but it seems to make things work?
            ** 0.4
            * u**2.4
            / (1 + 0.5 * u**2)
        )

        new_t_deadline = new_t_desired * 2  # Arbitrary, from the slides.

        return EnvState(
            state.target_positions,
            state.target_singer_sigmas,
            state.target_singer_thetas,
            state.ship_state,
            state.ship_idle_sensor,
            state.ship_sensor_cooldowns.at[state.ship_idle_sensor].set(
                new_t_dwell_estimate
            ),  # This probably shouldn't be recycled this way
            state.ship_searchers,
            state.tracker_t_desireds.at[track_index].set(new_t_desired),
            state.tracker_t_deadlines.at[track_index].set(new_t_deadline),
            state.tracker_t_dwell_estimates.at[track_index].set(new_t_dwell_estimate),
        )

    def reward(self, state, state_prime, action, params):
        reward = (action == 0) * jnp.min(
            jnp.array(
                [
                    0,
                    jnp.min(state.ship_searchers)
                    * params.search_penalty_per_second_late,
                ]
            )
        ) + (action != 0) * jnp.min(
            jnp.array(
                [
                    0,
                    state.tracker_t_desireds[action - SENSORS]
                    * params.track_penalty_per_second_late,
                ]
            )
        )  # TODO: handle dropping tasks,
        # maybe add drop penalty when something is not in state_prime?
        return reward

    def spawn_targets(self, key, params):
        # Okay, for now everything is some hypbtarget. To do the 'varying number of targets'
        # I think I need a big params.max_targets buffer and a "add_target(rng, target_index, targbuf, params)"
        # type function that goes and injects a new target into the buffer.
        # Or maybe I need to just generate a whole array of type 1s, 2s, and 3s and then just copy them into
        # the main buffer by multiplying with "frequency of 1, frequency of 2, frequency of 3" type math?
        # In my head, at the steady state, this looks like a long buffer of mostly zeroes with targets sprinkled in.
        # I'd then just disambiguate the targets by their positions, this means that filters will also be length params.max_targets.

        key_position, key_sigmas, key_thetas, key = jax.random.split(key, 4)

        target_positions = jax.random.uniform(
            key_position,
            (MAX_TARGETS, 9),
            minval=jnp.array(
                [
                    params.arena_min_dx,
                    -params.target_max_v,
                    -params.target_max_a,
                    params.arena_min_dy,
                    -params.target_max_v,
                    -params.target_max_a,
                    params.arena_min_dz,
                    # -params.target_max_v,
                    # -params.target_max_v / 10,
                    0,  # Targets spawn doing level flight
                    -params.target_max_a,
                ],
                dtype=DTYPE_FLOAT,
            ),
            maxval=jnp.array(
                [
                    params.arena_max_dx,
                    params.target_max_v,
                    params.target_max_a,
                    params.arena_max_dy,
                    params.target_max_v,
                    params.target_max_a,
                    params.arena_max_dz,
                    # params.target_max_v,
                    # params.target_max_v / 10,
                    0,  # Targets spawn doing level flight
                    params.target_max_a,
                ],
                dtype=DTYPE_FLOAT,
            ),
        )

        # Table III "Singer Manoeuvre Parameters for Three Target Types"
        # From A. Charlish, K. Woodbridge, and H. Griffiths,
        # ‘Phased array radar resource management using continuous double auction’,
        # IEEE Transactions on Aerospace and Electronic Systems,
        # vol. 51, no. 3, pp. 2212–2224, 2015.
        # DOI. No. 10.1109/TAES.2015.130558.
        #
        # Type 1: BigSigma in [20,35], BigTheta in [10,20]
        # Type 2: BigSigma in [0,5], BigTheta in [1,4]
        # Type 3: BigSigma in [5,20], BigTheta in [30,50]
        #
        # For simplicity, I'm using these "should encompass all" types.
        # Obvious TODO: implement the three types just by doing some "Minvals and Maxvals".
        # Less obvious TODO: specify the number of each type in the params.
        # Least obvious TODO: specify a scenario where truly different kinematics can coexist.
        target_singer_sigmas = jax.random.uniform(
            key_sigmas, (MAX_TARGETS, 1), dtype=DTYPE_FLOAT, minval=0, maxval=35
        )
        target_singer_thetas = jax.random.uniform(
            key_thetas, (MAX_TARGETS, 1), dtype=DTYPE_FLOAT, minval=1, maxval=50
        )

        return target_positions, target_singer_sigmas, target_singer_thetas

    def reset(
        self,
        key: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState]:
        """Environment-specific reset."""

        # targets have x,y,z,vx,vy,vz,ax,ay,az in m, m/s, m/s^2

        # target_positions = jnp.zeros((MAX_TARGETS, 9), dtype=DTYPE_FLOAT)
        # target_singer_sigmas = jnp.zeros((MAX_TARGETS, 1), dtype=DTYPE_FLOAT)
        # target_singer_thetas = jnp.zeros((MAX_TARGETS, 1), dtype=DTYPE_FLOAT)

        key_spawn_targets, key = jax.random.split(key)

        target_positions, target_singer_sigmas, target_singer_thetas = (
            self.spawn_targets(key_spawn_targets, params)
        )

        # Average time to leave the arena is 184_000m / 500m/s = 368s, so 0.05*368 = 18.4 targets
        # Mask off all but the first 20 targets
        first_n_targets = 20
        target_positions = target_positions.at[20:].set(
            jnp.zeros((MAX_TARGETS - first_n_targets, 9), dtype=DTYPE_FLOAT)
        )
        target_singer_sigmas = target_singer_sigmas.at[20:].set(
            jnp.zeros((MAX_TARGETS - first_n_targets, 1), dtype=DTYPE_FLOAT)
        )
        target_singer_thetas = target_singer_thetas.at[20:].set(
            jnp.zeros((MAX_TARGETS - first_n_targets, 1), dtype=DTYPE_FLOAT)
        )

        # PLATFORM PLATFORM PLATFORM PLATFORM PLATFORM PLATFORM PLATFORM PLATFORM

        # First build a ship
        ship_state = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=DTYPE_FLOAT)

        # Add sensors to the ship

        # This is a hack,
        # it causes the first step to timewarp 42 seconds into the future
        # This way hopefully some targets will spawn
        ship_sensor_cooldowns = (
            jnp.zeros(shape=(SENSORS,), dtype=DTYPE_FLOAT).at[0].set(42.0)
        )

        # searcher has t_desired for each wedge in seconds
        # ship_searchers = jnp.ones(
        ship_searchers = jnp.zeros(AZ_SLICES * EL_SLICES, dtype=DTYPE_FLOAT)

        # trackers have t_desired, t_deadline, t_dwell_estimate, in seconds
        ship_trackers = jnp.zeros((MAX_TARGETS, 3), dtype=DTYPE_FLOAT)

        # targets: jax.Array  # Yikes, not easy at all to implement different targets with different kinematics.
        # ship_state: jax.Array  # should we get this puppy moving? for now, origin
        # ship_sensors: jax.Array  # busy state of each sensor?
        # ship_searchers: jax.Array  # so this is the wedge countdown?
        # ship_trackers: jax.Array  # so this will be max_targets times timers array? I might need to be writing down target keys too...
        #
        tracker_t_desireds = jnp.zeros(MAX_TARGETS, dtype=DTYPE_FLOAT)
        tracker_t_deadlines = jnp.zeros(MAX_TARGETS, dtype=DTYPE_FLOAT)
        tracker_t_dwell_estimates = jnp.zeros(MAX_TARGETS, dtype=DTYPE_FLOAT)
        # TODO: divide by n_targets

        state = EnvState(
            target_positions=target_positions,
            target_singer_sigmas=target_singer_sigmas,
            target_singer_thetas=target_singer_thetas,
            ship_state=ship_state,
            ship_idle_sensor=0,
            ship_sensor_cooldowns=ship_sensor_cooldowns,
            ship_searchers=ship_searchers,
            tracker_t_desireds=tracker_t_desireds,
            tracker_t_deadlines=tracker_t_deadlines,
            tracker_t_dwell_estimates=tracker_t_dwell_estimates,
        )

        obskey, key = jax.random.split(key)
        return self.get_obs(state, params, obskey), state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int,  # Next task to be scheduled
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Examples implement Auto-Reset in here, but I'm not going to do that.
        # Due to rare and expensive resets, check out 'optimistic resets' from craftax.

        info = dict()

        state_after_action = jax.lax.cond(
            action == 0,  # The 0 action is reserved for search
            self.execute_search,
            self.execute_track,
            state,
            action,
            params,
        )

        # Find time until the next sensor is free, walk that far into the future.
        next_sensor_index = jnp.argmin(state_after_action.ship_sensor_cooldowns)
        delta_t = state_after_action.ship_sensor_cooldowns[next_sensor_index]

        key_target_update, key_poisson, key_spawn, key = jax.random.split(key, 4)

        proposed_target_positions = target_update(
            state_after_action.target_positions,
            state_after_action.target_singer_sigmas,
            state_after_action.target_singer_thetas,
            key_target_update,
            delta_t,
        )
        # Kill off any targets that have left the arena
        target_in_arena = (
            (proposed_target_positions[:, 0] >= params.arena_min_dx)
            & (proposed_target_positions[:, 0] < params.arena_max_dx)
            & (proposed_target_positions[:, 3] >= params.arena_min_dy)
            & (proposed_target_positions[:, 3] <= params.arena_max_dy)
            & (proposed_target_positions[:, 6] >= params.arena_min_dz)
            & (proposed_target_positions[:, 6] <= params.arena_max_dz)
            & (
                proposed_target_positions[:, 0]
                + proposed_target_positions[:, 3]
                + proposed_target_positions[:, 6]
                != 0  # This is because I'm using the origin as a parking spot for dead targets.
            )
        )

        proposed_positions_in_arena = jnp.where(
            target_in_arena[:, None],  # Reshape for broadcasting
            proposed_target_positions,  # Keep original positions where True
            jnp.zeros_like(proposed_target_positions),  # Zero out where False
        )

        inactive_targets = jnp.sum(proposed_positions_in_arena, axis=1) == 0

        # Spawn new targets

        new_targets = inactive_targets & (
            jax.random.poisson(
                key_poisson,
                lam=params.new_target_rate_parameter * delta_t / MAX_TARGETS,
                shape=(MAX_TARGETS,),
                dtype=DTYPE_INT,
            )
            > 0
        )

        spawn_positions, spawn_sigmas, spawn_thetas = self.spawn_targets(
            key_spawn, params
        )

        new_positions = jnp.where(
            new_targets[:, None], spawn_positions, proposed_positions_in_arena
        )
        new_sigmas = jnp.where(
            new_targets[:, None], spawn_sigmas, state_after_action.target_singer_sigmas
        )
        new_thetas = jnp.where(
            new_targets[:, None], spawn_thetas, state_after_action.target_singer_thetas
        )

        new_ship_state = state_after_action.ship_state
        new_idle_sensor = next_sensor_index
        new_ship_sensor_cooldowns = state_after_action.ship_sensor_cooldowns - delta_t
        new_ship_searchers = state_after_action.ship_searchers - delta_t

        new_tracker_t_desireds = jax.nn.relu(
            state_after_action.tracker_t_desireds - delta_t
        )  # hack to make output pretty, should really be multiplied by some "is active" mask.
        new_tracker_t_deadlines = state_after_action.tracker_t_deadlines - delta_t
        new_t_dwell_estimates = state_after_action.tracker_t_dwell_estimates  # same

        new_state = EnvState(
            new_positions,
            new_sigmas,
            new_thetas,
            new_ship_state,
            new_idle_sensor,
            new_ship_sensor_cooldowns,
            new_ship_searchers,
            new_tracker_t_desireds,
            new_tracker_t_deadlines,
            new_t_dwell_estimates,
        )

        return (
            self.get_obs(new_state, params, key),
            new_state,
            self.reward(state, new_state, action, params),
            self.is_terminal(state, params),
            info,
        )

    # figure out how long it will take until the next sensor is free
    #
    # use that time to update the state of the environment
    # use that time to update the state of the platform
    # I think reward falls out of this platform update? We see which tasks were dropped at least, unclear if we can see which ones improved.
    #
    # generate a whole new collection of requests
    #
    # # slides say reward is 1/cost, but this gives frequent 1/0. Strange choice.
    #     reward = -cost  # slides say divided? I get divide by zero nonsense.
    #     # Also, I'm only getting reward from the task I just completed.
    #     # Maybe eventually also take cost from dropping?
    #
    def get_obs(
        self,
        state,
        params,
        key,
        # ) -> jax.Array:
    ) -> dict[str, jax.Array]:
        """Applies observation function to state."""
        return {
            "sensor": state.ship_idle_sensor,
            "search": {
                "t_desired": state.ship_searchers,
            },
            "trackers": {
                "t_desired": state.tracker_t_desireds,
                "t_deadline": state.tracker_t_deadlines,
                "t_dwell": state.tracker_t_dwell_estimates,
            },
        }

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state transition is terminal."""
        # go find tag-place-to-enable-reset and uncomment the auto-reset logic below this
        # For now as long as I'm never auto-resetting, I've stepped over it.
        # return jnp.array(False, dtype=bool)
        return False

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        # return 1 ? or maybe 'max trackers'?
        raise NotImplementedError

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Discrete(SENSORS + MAX_TARGETS)

    # TODO: @amos update this to reflect the actual observation space
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""

        return spaces.Dict(
            {
                "sensor": spaces.Discrete(SENSORS),
                "search": spaces.Dict(
                    {
                        "t_start": spaces.Box(
                            0, jnp.finfo(DTYPE_FLOAT).max, shape=(1,), dtype=DTYPE_FLOAT
                        ),
                        "t_elapsed": spaces.Box(
                            0,
                            jnp.finfo(DTYPE_FLOAT).max,
                            (AZ_SLICES * EL_SLICES,),
                            DTYPE_FLOAT,
                        ),
                    }
                ),
                "trackers": spaces.Dict(
                    {
                        "on": spaces.Box(0, 1, (MAX_TARGETS,), DTYPE_INT),
                        "t_start": spaces.Box(
                            0,
                            jnp.finfo(DTYPE_FLOAT).max,
                            (MAX_TARGETS,),
                            DTYPE_FLOAT,
                        ),
                        "t_dwell": spaces.Box(
                            0,
                            jnp.finfo(DTYPE_FLOAT).max,
                            (MAX_TARGETS,),
                            DTYPE_FLOAT,
                        ),
                        # "delay_cost", "drop_cost","deadline_time",
                    }
                ),
            }
        )

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        raise NotImplementedError


def target_update(
    target_positions: jax.Array,
    target_maneuver_standard_deviation: jax.Array,  # "Big Sigma"
    target_maneuver_time_constants: jax.Array,  # "Big Theta"
    key: jax.Array,
    t: float,
) -> jax.Array:
    """State transition matrix"""
    # state must be [x,vx,ax, y,vy,ay, z,vz,az]
    newton_block = jnp.array(
        [
            [1, 0, 0],
            [t, 1, 0],
            [t**2 / 2, t, 1],
        ],
        dtype=DTYPE_FLOAT,
    )
    newton_matrix = block_diag(newton_block, newton_block, newton_block)

    # This will create an (n_targets, 9) matrix where each row follows the pattern [1, 1, rho, 1, 1, rho, 1, 1, rho]
    rho = jnp.exp(-t / target_maneuver_time_constants) * jnp.array(
        # [0, 0, 1, 0, 0, 1, 0, 0, 1]  # prevent nosedives with [0 0 1 0 0 1 0 0 0.1] ?
        # [0, 0, 1, 0, 0, 1, 0, 0, 0] # maintain level flight?
        [0, 0, 1, 0, 0, 1, 0, 0, 0.1]
    ) + jnp.array([1, 1, 0, 1, 1, 0, 1, 1, 0])

    acceleration_noise = (
        jnp.sqrt((1 - rho**2))
        * jax.random.normal(key, shape=target_positions.shape, dtype=DTYPE_FLOAT)
        * target_maneuver_standard_deviation
    )

    return (target_positions @ newton_matrix) * rho + acceleration_noise
