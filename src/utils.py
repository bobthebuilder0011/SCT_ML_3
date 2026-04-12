import yaml
import logging
import os

def load_config(config_path='config/config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_level=logging.INFO):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")
