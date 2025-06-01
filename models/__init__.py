import importlib
from typing import Dict, Type

# Dictionary to store model class names and their module paths
_MODEL_REGISTRY: Dict[str, str] = {
    'LSTM': '.lstm',
    'TCN': '.tcn',
    'Transformer': '.transformer',
    'HybridTCNLSTM': '.hybrid_tcn_lstm',
    'PatchTST': '.patch_tst',
    'MLP': '.mlp'
}

def __getattr__(name: str):
    """Lazy import of model classes."""
    if name in _MODEL_REGISTRY:
        module = importlib.import_module(_MODEL_REGISTRY[name], package='models')
        return getattr(module, name)
    raise AttributeError(f"module 'models' has no attribute '{name}'")

__all__ = list(_MODEL_REGISTRY.keys()) 