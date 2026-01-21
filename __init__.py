# -*- coding: utf-8 -*-


# 1. Configuration
from .config import Config

# 2. Core Entities (Client & Server)
from .client import FedClient
from .server import (
    FedCHGServer,
    calculate_global_prototype,
    aggregate_structural_weighted
)

# 3. Models (Neural Networks)
from .models import (
    FedCHG_GAE,
    FeatureModel,
    StructuralModel,
    FedCHG_Classifier
)

# 4. Data Handling
# [FIXED] Corrected imports to match data_loader.py definitions
from .data_loader import (
    CrossDomainDataLoader,
    load_cross_domain_data
)

# 5. Feature Extraction
from .features import get_se_stru

# 6. Utilities
from .utils import (
    setup_logger,
    compute_structural_similarity,
    compute_gradient_influence
)

__all__ = [
    'Config',
    'FedClient',
    'FedCHGServer',
    'calculate_global_prototype',
    'aggregate_structural_weighted',
    'FedCHG_GAE',
    'FeatureModel',
    'StructuralModel',
    'FedCHG_Classifier',
    'CrossDomainDataLoader',
    'load_cross_domain_data',
    'get_se_stru',
    'setup_logger',
    'compute_structural_similarity',
    'compute_gradient_influence'
]