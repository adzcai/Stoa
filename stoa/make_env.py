import dataclasses
import functools
from typing import Any, Callable, ParamSpec, Tuple

from colorama import Fore, Style
from stoa import (
    AddStartFlagAndPrevAction,
    AutoResetWrapper,
    Environment,
    EpisodeStepLimitWrapper,
    MultiDiscreteSpace,
    MultiDiscreteToDiscreteWrapper,
    ObservationExtractWrapper,
    RecordEpisodeMetrics,
)
from stoa.core_wrappers.auto_reset import CachedAutoResetWrapper
from stoa.core_wrappers.optimistic_auto_reset import OptimisticResetVmapWrapper
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.core_wrappers.wrapper import AddRNGKey
from stoa.utility_wrappers.extras_transforms import NoExtrasWrapper

from stoa.env_factory import EnvFactory
from stoa.env_adapters.jax_to_factory import JaxEnvFactory


def env_maker(
    make_env: Callable[..., Tuple[Environment, Environment]],
) -> Callable[..., Tuple[Environment, Environment]]:
    """Applies core wrappers for JAX-based environments and any user-defined optional wrappers from the configuration.

    This includes wrappers for:
    - Adding an RNG key to the environment state.
    - Automatically resetting episodes upon termination.
    - Recording episode metrics (return, length, etc.).
    - Vectorizing the environment for batched execution.

    Args:
        env: The environment to wrap.
        config: The system configuration.

    Returns:
        A tuple containing the training and evaluation environments.
    """

    @functools.wraps(make_env)
    def wrapper(
        *args,
        wrapper_fn: Callable[[Environment], Environment],
        use_optimistic_reset: bool = False,
        num_envs: int | None = None,
        reset_ratio: int = 16,
        use_cached_auto_reset: bool = False,
        **kwargs,
    ) -> Tuple[Environment, Environment]:
        env, eval_env = make_env(*args, **kwargs)

        if wrapper_fn is not None:
            env = wrapper_fn(env)
            eval_env = wrapper_fn(eval_env)

        env = AddRNGKey(env)
        env = RecordEpisodeMetrics(env)

        if use_optimistic_reset:
            if num_envs is None:
                raise ValueError("num_envs must be specified when using optimistic reset.")
            env = OptimisticResetVmapWrapper(
                env,
                num_envs,
                min(reset_ratio, num_envs),
                next_obs_in_extras=True,
            )
        else:
            if use_cached_auto_reset:
                env = CachedAutoResetWrapper(env, next_obs_in_extras=True)
            else:
                env = AutoResetWrapper(env, next_obs_in_extras=True)
            env = VmapWrapper(env)

        return env, eval_env

    return wrapper


@env_maker
def make_jumanji_env(
    scenario_name: str,
    multi_agent: bool = False,
    observation_attribute: str | None = None,
    **env_kwargs,
) -> Tuple[Environment, Environment]:
    """Creates and wraps a Jumanji environment."""
    import jumanji
    import jumanji.wrappers as jumanji_wrappers
    from stoa.env_adapters.jumanji import JumanjiToStoa

    env = jumanji.make(scenario_name, **env_kwargs)
    eval_env = jumanji.make(scenario_name, **env_kwargs)

    if multi_agent:
        env = jumanji_wrappers.MultiToSingleWrapper(env)
        eval_env = jumanji_wrappers.MultiToSingleWrapper(eval_env)

    if observation_attribute is None:
        raise ValueError("observation_attribute must be specified for Jumanji environments.")

    env = JumanjiToStoa(env)
    env = ObservationExtractWrapper(env, observation_attribute)
    eval_env = JumanjiToStoa(eval_env)
    eval_env = ObservationExtractWrapper(eval_env, observation_attribute)

    if isinstance(env.action_space(), MultiDiscreteSpace):
        env = MultiDiscreteToDiscreteWrapper(env)
        eval_env = MultiDiscreteToDiscreteWrapper(eval_env)

    return env, eval_env


def _create_gymnax_env_instance(
    scenario_name: str,
    env_kwargs: dict,
    env_make_fn: Callable[[str], Tuple[Any, Any]],
) -> Tuple[Any, Any]:
    """Instantiates a Gymnax-like environment, handling init and param kwargs."""
    _, default_params = env_make_fn(scenario_name)
    param_fields = {f.name for f in dataclasses.fields(default_params)}

    init_kwargs = {k: v for k, v in env_kwargs.items() if k not in param_fields}
    params_kwargs = {k: v for k, v in env_kwargs.items() if k in param_fields}

    env, env_params = env_make_fn(scenario_name, **init_kwargs)
    if params_kwargs:
        env_params = dataclasses.replace(env_params, **params_kwargs)
    return env, env_params


@env_maker
def make_gymnax_env(scenario_name: str, **env_kwargs) -> Tuple[Environment, Environment]:
    """Creates and wraps a Gymnax environment."""
    import gymnax
    from stoa.env_adapters.gymnax import GymnaxToStoa

    env, env_params = _create_gymnax_env_instance(scenario_name, env_kwargs, gymnax.make)
    eval_env, eval_env_params = _create_gymnax_env_instance(scenario_name, env_kwargs, gymnax.make)

    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


@env_maker
def make_popgym_arcade_env(scenario_name: str, **env_kwargs) -> Tuple[Environment, Environment]:
    """Creates and wraps a PopGym Arcade environment."""
    import popgym_arcade
    from stoa.env_adapters.gymnax import GymnaxToStoa

    env, env_params = _create_gymnax_env_instance(scenario_name, env_kwargs, popgym_arcade.make)
    eval_env, eval_env_params = _create_gymnax_env_instance(
        scenario_name, env_kwargs, popgym_arcade.make
    )

    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


@env_maker
def make_xland_minigrid_env(scenario_name: str, **env_kwargs) -> Tuple[Environment, Environment]:
    """Creates and wraps an XLand-MiniGrid environment."""
    import xminigrid
    from stoa.env_adapters.xminigrid import XMiniGridToStoa

    env, env_params = xminigrid.make(scenario_name, **env_kwargs)
    eval_env, eval_env_params = xminigrid.make(scenario_name, **env_kwargs)

    env = XMiniGridToStoa(env, env_params)
    eval_env = XMiniGridToStoa(eval_env, eval_env_params)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


@env_maker
def make_brax_env(scenario_name: str, **env_kwargs) -> Tuple[Environment, Environment]:
    """Creates and wraps a Brax environment."""
    from brax.envs import create as brax_make
    from stoa.env_adapters.brax import BraxToStoa

    env = brax_make(scenario_name, auto_reset=False, **env_kwargs)
    eval_env = brax_make(scenario_name, auto_reset=False, **env_kwargs)

    env = BraxToStoa(env)
    eval_env = BraxToStoa(eval_env)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


@env_maker
def make_kinetix_env(
    scenario_name: str,
    dense_reward_scale: float,
    frame_skip: int,
    scenario: Any,
    kinetix: Any,
) -> Tuple[Environment, Environment]:
    """Creates and wraps a Kinetix environment."""
    import jax
    from kinetix.environment import EnvState, StaticEnvParams, make_kinetix_env
    from kinetix.environment.env import KinetixEnv
    from kinetix.environment.ued.ued import make_reset_fn_sample_kinetix_level
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.util.config import generate_params_from_config
    from kinetix.util.saving import load_evaluation_levels
    from stoa.env_adapters.kinetix import KinetixToStoa

    env_params, override_static_env_params = generate_params_from_config(
        dict(kinetix.env_size)
        | {
            "dense_reward_scale": dense_reward_scale,
            "frame_skip": frame_skip,
        }
    )

    def _get_static_params_and_reset_fn(
        level_config,
    ) -> tuple[Callable, StaticEnvParams]:
        if level_config.mode == "list":
            levels = level_config.levels
            levels_to_reset_to, static_env_params = load_evaluation_levels(levels)

            def reset(rng: jax.Array) -> EnvState:
                rng, _rng = jax.random.split(rng)
                level_idx = jax.random.randint(_rng, (), 0, len(levels))
                sampled_level = jax.tree.map(lambda x: x[level_idx], levels_to_reset_to)
                return sampled_level

        elif level_config.mode == "random":
            return (
                make_reset_fn_sample_kinetix_level(env_params, override_static_env_params),
                override_static_env_params,
            )
        else:
            raise ValueError(f"Unsupported level mode: {level_config.mode}")
        return reset, static_env_params

    reset_fn_train, static_env_params_train = _get_static_params_and_reset_fn(kinetix.train)
    reset_fn_eval, static_env_params_eval = _get_static_params_and_reset_fn(kinetix.eval)

    def _make_env(reset_fn: Callable, static_env_params: StaticEnvParams) -> KinetixEnv:
        env = make_kinetix_env(
            action_type=ActionType.from_string(scenario.action_type),
            observation_type=ObservationType.from_string(scenario.observation_type),
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=False,
        )
        return KinetixToStoa(env, env_params)

    env = _make_env(reset_fn=reset_fn_train, static_env_params=static_env_params_train)
    eval_env = _make_env(reset_fn=reset_fn_eval, static_env_params=static_env_params_eval)

    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


@env_maker
def make_craftax_env(scenario_name: str) -> Tuple[Environment, Environment]:
    """Creates and wraps a Crafter (Craftax) environment."""
    from craftax.craftax_env import make_craftax_env_from_name
    from stoa.env_adapters.gymnax import GymnaxToStoa

    env = make_craftax_env_from_name(scenario_name, auto_reset=False)
    eval_env = make_craftax_env_from_name(scenario_name, auto_reset=False)
    env_params = env.default_params
    eval_env_params = eval_env.default_params

    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


@env_maker
def make_debug_env(scenario_name: str, **env_kwargs) -> Tuple[Environment, Environment]:
    """Creates a simple debug environment for testing purposes."""
    from stoa.debug_env import DEBUG_ENVIRONMENTS

    env = DEBUG_ENVIRONMENTS[scenario_name](**env_kwargs)
    eval_env = DEBUG_ENVIRONMENTS[scenario_name](**env_kwargs)
    return env, eval_env


@env_maker
def make_popjym_env(scenario_name: str, **env_kwargs) -> Tuple[Environment, Environment]:
    """Creates and wraps a POPJym environment."""
    import popjym
    from stoa.env_adapters.gymnax import GymnaxToStoa

    env, env_params = popjym.make(scenario_name, **env_kwargs)
    eval_env, eval_env_params = popjym.make(scenario_name, **env_kwargs)

    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    env = AddStartFlagAndPrevAction(env)
    eval_env = AddStartFlagAndPrevAction(eval_env)
    return env, eval_env


@env_maker
def make_navix_env(scenario_name: str, **env_kwargs) -> Tuple[Environment, Environment]:
    """Creates and wraps a Navix environment."""
    import navix
    from stoa.env_adapters.navix import NavixToStoa

    env = navix.make(scenario_name, **env_kwargs)
    eval_env = navix.make(scenario_name, **env_kwargs)

    env = NavixToStoa(env)
    eval_env = NavixToStoa(eval_env)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


@env_maker
def make_playground_env(
    scenario_name: str,
    use_default_domain_randomizer: bool = False,
    max_episode_steps: int = 1000,
    **env_kwargs,
) -> Tuple[Environment, Environment]:
    """Creates and wraps a MuJoCo Playground environment."""
    import mujoco_playground
    from stoa.env_adapters.playground import MuJoCoPlaygroundToStoa

    env_cfg = mujoco_playground.registry.get_default_config(scenario_name)
    env = mujoco_playground.registry.load(
        scenario_name, config=env_cfg, config_overrides=env_kwargs
    )
    eval_env = mujoco_playground.registry.load(
        scenario_name, config=env_cfg, config_overrides=env_kwargs
    )

    domain_randomizer_fn = None
    if use_default_domain_randomizer:
        domain_randomizer_fn = mujoco_playground.registry.get_domain_randomizer(scenario_name)
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Using default domain randomizer for "
            f"'{scenario_name}': {domain_randomizer_fn.__name__}.{Style.RESET_ALL}"
        )

    env = MuJoCoPlaygroundToStoa(env, domain_randomizer_fn=domain_randomizer_fn)
    eval_env = MuJoCoPlaygroundToStoa(eval_env, domain_randomizer_fn=domain_randomizer_fn)
    env = EpisodeStepLimitWrapper(env, max_episode_steps)
    eval_env = EpisodeStepLimitWrapper(eval_env, max_episode_steps)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    return env, eval_env


# A dispatcher mapping environment suite names to their respective maker functions.
ENV_MAKERS = {
    "jumanji": make_jumanji_env,
    "gymnax": make_gymnax_env,
    "brax": make_brax_env,
    "craftax": make_craftax_env,
    "popgym_arcade": make_popgym_arcade_env,
    "xland_minigrid": make_xland_minigrid_env,
    "popjym": make_popjym_env,
    "navix": make_navix_env,
    "kinetix": make_kinetix_env,
    "mujoco_playground": make_playground_env,
    "debug": make_debug_env,
}


def make(suite_name: str, scenario_name: str, **kwargs) -> Tuple[Environment, Environment]:
    """Creates training and evaluation environments based on the provided configuration.

    This function uses a dispatcher to call the correct maker function for the
    specified environment suite (e.g., 'jumanji', 'brax'). This approach enables
    lazy loading of heavy dependencies, improving startup time.

    Args:
        config: The system configuration, which must specify `env.env_name`
            and `env.scenario.name`.

    Returns:
        A tuple containing the instantiated training and evaluation environments.
    """

    if suite_name not in ENV_MAKERS:
        raise ValueError(
            f"Unsupported environment suite '{suite_name}'. "
            f"Available suites: {list(ENV_MAKERS.keys())}"
        )

    maker_function = ENV_MAKERS[suite_name]
    envs = maker_function(scenario_name, **kwargs)

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Created environments for Suite: {suite_name} - "
        f"Scenario: {scenario_name}{Style.RESET_ALL}"
    )
    return envs


def make_factory(
    suite_name: str,
    scenario_name: str,
    seed: int,
    apply_wrapper_fn: Callable = lambda x: x,
    **kwargs,
) -> EnvFactory:
    """Creates a factory for generating environments.

    This is used for systems that require an environment factory rather than
    pre-instantiated environments, such as those using non-JAX environments
    like Gymnasium or EnvPool.

    Args:
        config: The system configuration.

    Returns:
        An `EnvFactory` instance.
    """

    if suite_name == "envpool":
        from stoa.env_factory import EnvPoolFactory

        return EnvPoolFactory(
            scenario_name,
            init_seed=seed,
            apply_wrapper_fn=apply_wrapper_fn,
            **kwargs,
        )
    elif suite_name == "gymnasium":
        from stoa.env_factory import GymnasiumFactory

        return GymnasiumFactory(
            scenario_name,
            init_seed=seed,
            apply_wrapper_fn=apply_wrapper_fn,
            **kwargs,
        )
    else:
        # For all other JAX-based environments, create a single instance
        # and wrap it in a JaxEnvFactory.
        train_env = make(suite_name, scenario_name, **kwargs)[0]
        return JaxEnvFactory(train_env, init_seed=seed, apply_wrapper_fn=apply_wrapper_fn)
