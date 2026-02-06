"""Configuration management for EdgeMemory."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


_config: Optional[Dict[str, Any]] = None


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    global _config
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, 'r') as f:
        _config = yaml.safe_load(f)
    return _config


def get_config() -> Dict[str, Any]:
    """Get current configuration, loading defaults if needed."""
    global _config
    if _config is None:
        try:
            _config = load_config()
        except FileNotFoundError:
            _config = _default_config()
    return _config


def get(key: str, default: Any = None) -> Any:
    """Get a config value by dot-separated key. E.g. 'llm.model'."""
    cfg = get_config()
    keys = key.split('.')
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def _default_config() -> Dict[str, Any]:
    return {
        "llm": {"model": "phi3", "temperature": 0.7, "max_tokens": 1024},
        "asr": {"model_size": "base", "device": "cpu", "compute_type": "int8"},
        "embeddings": {"provider": "ollama", "model": "nomic-embed-text", "dimension": 768},
        "storage": {
            "db_path": "data/memories.db",
            "vector_path": "data/vectors/index",
            "graph_path": "data/knowledge.pkl",
        },
        "retrieval": {
            "dense_k": 20, "sparse_k": 20, "graph_k": 10, "sql_k": 20,
            "fusion_weights": {"dense": 0.4, "sparse": 0.2, "graph": 0.2, "sql": 0.2},
            "rrf_k": 60,
        },
        "server": {"host": "0.0.0.0", "port": 8000, "debug": False},
        "classifiers": {"confidence_threshold": 0.7, "use_lightweight": True},
    }
