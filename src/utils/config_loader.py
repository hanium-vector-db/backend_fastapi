import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """Configuration loader for backend and UI settings"""
    
    def __init__(self):
        self.backend_config = None
        self.ui_config = None
        self._load_configs()
    
    def _load_configs(self):
        """Load both backend and UI configuration files"""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load backend config
        backend_config_path = os.path.join(project_root, 'config', 'backend_config.yaml')
        if os.path.exists(backend_config_path):
            with open(backend_config_path, 'r', encoding='utf-8') as f:
                self.backend_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Backend config file not found: {backend_config_path}")
        
        # Load UI config
        ui_config_path = os.path.join(project_root, 'config', 'ui_config.yaml')
        if os.path.exists(ui_config_path):
            with open(ui_config_path, 'r', encoding='utf-8') as f:
                self.ui_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"UI config file not found: {ui_config_path}")
    
    def get_backend_config(self, section: str = None, key: str = None) -> Any:
        """Get backend configuration value"""
        if section is None:
            return self.backend_config
        
        if section not in self.backend_config:
            raise KeyError(f"Section '{section}' not found in backend config")
        
        if key is None:
            return self.backend_config[section]
        
        if key not in self.backend_config[section]:
            raise KeyError(f"Key '{key}' not found in section '{section}' of backend config")
        
        return self.backend_config[section][key]
    
    def get_ui_config(self, section: str = None, key: str = None) -> Any:
        """Get UI configuration value"""
        if section is None:
            return self.ui_config
        
        if section not in self.ui_config:
            raise KeyError(f"Section '{section}' not found in UI config")
        
        if key is None:
            return self.ui_config[section]
        
        if key not in self.ui_config[section]:
            raise KeyError(f"Key '{key}' not found in section '{section}' of UI config")
        
        return self.ui_config[section][key]
    
    # Convenient property methods for commonly used configs
    
    @property
    def server_host(self) -> str:
        return self.get_backend_config('server', 'host')
    
    @property
    def server_port(self) -> int:
        return self.get_backend_config('server', 'port')
    
    @property
    def database_config(self) -> Dict[str, Any]:
        return self.get_backend_config('database')
    
    @property
    def mariadb_config(self) -> Dict[str, Any]:
        return self.get_backend_config('database', 'mariadb')
    
    @property
    def vector_db_path(self) -> str:
        return self.get_backend_config('database', 'vector_db_path')
    
    @property
    def faiss_db_path(self) -> str:
        return self.get_backend_config('database', 'faiss_db_path')
    
    @property
    def models_config(self) -> Dict[str, Any]:
        return self.get_backend_config('models')
    
    @property
    def default_llm_config(self) -> Dict[str, Any]:
        return self.get_backend_config('models', 'default_llm')
    
    @property
    def embedding_config(self) -> Dict[str, Any]:
        return self.get_backend_config('models', 'embedding')
    
    @property
    def model_cache_dir(self) -> str:
        return self.get_backend_config('models', 'cache_dir')
    
    @property
    def rag_config(self) -> Dict[str, Any]:
        return self.get_backend_config('rag')
    
    @property
    def external_web_rag_config(self) -> Dict[str, Any]:
        return self.get_backend_config('rag', 'external_web')
    
    @property
    def internal_db_rag_config(self) -> Dict[str, Any]:
        return self.get_backend_config('rag', 'internal_db')
    
    @property
    def ui_server_config(self) -> Dict[str, Any]:
        return self.get_ui_config('ui_server')
    
    @property
    def ui_backend_api_config(self) -> Dict[str, Any]:
        return self.get_ui_config('backend_api')
    
    @property
    def gradio_config(self) -> Dict[str, Any]:
        return self.get_ui_config('gradio')

# Global config loader instance
config = ConfigLoader()