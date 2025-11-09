"""
Configuration Management for NBA Prediction Model
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Example:
            config.get('data_collection.rate_limit.timeout', 120)
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
                
        return value if value is not None else default
    
    @property
    def seasons(self):
        """Get list of seasons to collect data for."""
        return self.get('data_collection.seasons', [])
    
    @property
    def rate_limit_interval(self):
        """Get rate limiting interval in seconds."""
        rps = self.get('data_collection.rate_limit.requests_per_second', 1.4)
        return 1.0 / rps
    
    @property
    def api_timeout(self):
        """Get API timeout in seconds."""
        return self.get('data_collection.rate_limit.timeout', 120)
    
    @property
    def api_headers(self):
        """Get custom headers for API requests."""
        return self.get('data_collection.headers', {})
    
    @property
    def target_threshold(self):
        """Get threshold for 'both teams lead' target."""
        return self.get('target.threshold', 5)
    
    @property
    def four_factors(self):
        """Get Dean Oliver's Four Factors."""
        return self.get('features.four_factors', [])
    
    @property
    def rolling_windows(self):
        """Get rolling window sizes for recent form."""
        return self.get('features.rolling_windows', [10, 20])
    
    @property
    def data_paths(self):
        """Get data directory paths."""
        base_path = Path(__file__).parent.parent.parent
        return {
            'raw': base_path / self.get('paths.data.raw', 'data/raw'),
            'processed': base_path / self.get('paths.data.processed', 'data/processed'),
            'labels': base_path / self.get('paths.data.labels', 'data/labels'),
        }
    
    @property
    def model_path(self):
        """Get models directory path."""
        base_path = Path(__file__).parent.parent.parent
        return base_path / self.get('paths.models', 'models')
    
    @property
    def output_path(self):
        """Get outputs directory path."""
        base_path = Path(__file__).parent.parent.parent
        return base_path / self.get('paths.outputs', 'outputs')


# Global config instance
_config = None

def get_config(config_path: str = None) -> Config:
    """Get global config instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
