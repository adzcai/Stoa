"""Stoa: A JAX-Native Interface for Reinforcement Learning Environments."""

# Version
__version__ = "0.1.1"

# Core wrappers
from stoa.core_wrappers.auto_reset import AutoResetWrapper
from stoa.core_wrappers.episode_metrics import (
    RecordEpisodeMetrics,
    get_final_step_metrics,
)
from stoa.core_wrappers.wrapper import AddRNGKey, StateWithKey, Wrapper, WrapperState

# Core types and abstractions
from stoa.env_types import (
    Action,
    ActionMask,
    Discount,
    EnvParams,
    Observation,
    Reward,
    State,
    StepCount,
    StepType,
    TimeStep,
    TimeStepExtras,
)
from stoa.environment import Environment

# Spaces
from stoa.spaces import (
    ArraySpace,
    BoundedArraySpace,
    DictSpace,
    DiscreteSpace,
    EnvironmentSpace,
    MultiDiscreteSpace,
    Space,
    TupleSpace,
    make_continuous,
    make_discrete,
)

# Action space wrappers
from stoa.utility_wrappers.action_space_transforms import (
    MultiBoundedToBoundedWrapper,
    MultiDiscreteToDiscreteWrapper,
)
from stoa.utility_wrappers.extras_transforms import ConsistentExtrasWrapper
from stoa.utility_wrappers.flatten_obs import FlattenObservationWrapper
from stoa.utility_wrappers.frame_stacking import FrameStackingWrapper

# Observation wrappers
from stoa.utility_wrappers.obs_extract import ObservationExtractWrapper
from stoa.utility_wrappers.obs_transforms import (
    AddActionMaskWrapper,
    AddStartFlagAndPrevAction,
    AddStepCountWrapper,
    MakeChannelLast,
    ObservationTypeWrapper,
)
from stoa.utility_wrappers.step_limit import EpisodeStepLimitWrapper

# Environment adapters (optional dependencies)
__all__ = [
    # Version
    "__version__",
    # Core types
    "Action",
    "ActionMask",
    "Discount",
    "EnvParams",
    "Observation",
    "Reward",
    "State",
    "StepCount",
    "StepType",
    "TimeStep",
    "TimeStepExtras",
    "Environment",
    # Spaces
    "ArraySpace",
    "BoundedArraySpace",
    "DictSpace",
    "DiscreteSpace",
    "EnvironmentSpace",
    "MultiDiscreteSpace",
    "Space",
    "TupleSpace",
    "make_continuous",
    "make_discrete",
    # Core wrappers
    "Wrapper",
    "WrapperState",
    "AddRNGKey",
    "StateWithKey",
    "AutoResetWrapper",
    "RecordEpisodeMetrics",
    "get_final_step_metrics",
    # Observation wrappers
    "ObservationExtractWrapper",
    "AddStartFlagAndPrevAction",
    "MakeChannelLast",
    "FlattenObservationWrapper",
    "FrameStackingWrapper",
    "AddStepCountWrapper",
    "AddActionMaskWrapper",
    "ObservationTypeWrapper",
    # Utility wrappers
    "EpisodeStepLimitWrapper",
    "ConsistentExtrasWrapper",
    # Action wrappers
    "MultiDiscreteToDiscreteWrapper",
    "MultiBoundedToBoundedWrapper",
]

try:
    from stoa.env_wrappers.brax import BraxToStoa

    __all__.append("BraxToStoa")
except ImportError:
    pass

try:
    from stoa.env_wrappers.gymnax import GymnaxToStoa

    __all__.append("GymnaxToStoa")
except ImportError:
    pass

try:
    from stoa.env_wrappers.jumanji import JumanjiToStoa

    __all__.append("JumanjiToStoa")
except ImportError:
    pass

try:
    from stoa.env_wrappers.kinetix import KinetixToStoa

    __all__.append("KinetixToStoa")
except ImportError:
    pass

try:
    from stoa.env_wrappers.navix import NavixToStoa

    __all__.append("NavixToStoa")
except ImportError:
    pass

try:
    from stoa.env_wrappers.pgx import PGXToStoa

    __all__.append("PGXToStoa")
except ImportError:
    pass

try:
    from stoa.env_wrappers.playground import MuJoCoPlaygroundToStoa

    __all__.append("MuJoCoPlaygroundToStoa")
except ImportError:
    pass

try:
    from stoa.env_wrappers.xminigrid import XMiniGridToStoa

    __all__.append("XMiniGridToStoa")
except ImportError:
    pass
