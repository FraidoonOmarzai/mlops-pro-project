"""
Configuration Manager Module
Handles loading and managing all configuration from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict
from src.logger import logger
from src.exception import CustomException
import sys


class ConfigManager:
    """
    Manages application configuration from YAML files.
    Provides methods to access different configuration sections.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ConfigManager and load configuration.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model_config = self._load_model_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load main configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
            
        Raises:
            CustomException: If configuration file cannot be loaded
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise CustomException(f"Configuration file not found: {self.config_path}", sys)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise CustomException(f"Error parsing YAML file: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_model_config(self) -> Dict[str, Any]:
        """
        Load model hyperparameters configuration.
        
        Returns:
            Dictionary containing model configurations
            
        Raises:
            CustomException: If model configuration file cannot be loaded
        """
        try:
            model_config_path = "config/model_config.yaml"
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            logger.info(f"Model configuration loaded from {model_config_path}")
            return model_config
        except FileNotFoundError:
            logger.error(f"Model configuration file not found: {model_config_path}")
            raise CustomException(f"Model configuration file not found: {model_config_path}", sys)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing model config YAML: {e}")
            raise CustomException(f"Error parsing model config YAML: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_ingestion_config(self) -> Dict[str, Any]:
        """
        Get data ingestion configuration.
        
        Returns:
            Dictionary with data ingestion settings
        """
        return self.config['data_ingestion']
    
    def get_data_validation_config(self) -> Dict[str, Any]:
        """
        Get data validation configuration.
        
        Returns:
            Dictionary with data validation settings
        """
        return self.config['data_validation']
    
    def get_data_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get data preprocessing configuration.
        
        Returns:
            Dictionary with preprocessing settings
        """
        return self.config['data_preprocessing']
    
    def get_model_training_config(self) -> Dict[str, Any]:
        """
        Get model training configuration.
        
        Returns:
            Dictionary with model training settings
        """
        return self.config['model_training']
    
    def get_model_evaluation_config(self) -> Dict[str, Any]:
        """
        Get model evaluation configuration.
        
        Returns:
            Dictionary with evaluation settings
        """
        return self.config['model_evaluation']
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """
        Get MLflow configuration.
        
        Returns:
            Dictionary with MLflow settings
        """
        return self.config['mlflow']
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of model hyperparameters
        """
        if model_name not in self.model_config:
            raise ValueError(f"Model {model_name} not found in model configuration")
        return self.model_config[model_name]
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        try:
            directories = [
                self.config['artifacts_root'],
                self.config['data_root'],
                'data/raw',
                'data/processed',
                'artifacts/models',
                'artifacts/preprocessors',
                'artifacts/metrics',
                'logs',
                'mlruns'
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                
            logger.info("All directories created successfully")
        except Exception as e:
            raise CustomException(e, sys)


# Global config instance - Import this in other modules
config_manager = ConfigManager()