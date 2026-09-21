"""TabPFN models and a predictive interface for relational tables."""

from relarena_core.userdb import (
    FittedPredictor,
    PredictiveContext,
    PredictiveQuery,
    PredictiveQuerySpec,
)

from tabpfn_rel.estimator import TabPFNRel
from tabpfn_rel.model import (
    TABPFN_REL_CLIENT_SPACE,
    TABPFN_REL_LOCAL_SPACE,
    TabPFNRelClientModel,
    TabPFNRelLocalModel,
    TabPFNRelModel,
)

__all__ = [
    "TABPFN_REL_CLIENT_SPACE",
    "TABPFN_REL_LOCAL_SPACE",
    "FittedPredictor",
    "PredictiveContext",
    "PredictiveQuery",
    "PredictiveQuerySpec",
    "TabPFNRel",
    "TabPFNRelClientModel",
    "TabPFNRelLocalModel",
    "TabPFNRelModel",
]
__version__ = "0.0.3"
