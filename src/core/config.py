from pydantic_settings import BaseSettings
from pydantic import model_validator, ConfigDict
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import config

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8',
        extra='ignore'  # .env의 추가 필드 무시
    )
    
    # 설정 파일에서 읽어온 모델 설정들
    @property
    def available_models(self) -> List[str]:
        return ["qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"]
    
    @property
    def selected_model(self) -> str:
        return config.default_llm_config['model_key']
    
    @property
    def database_url(self) -> str:
        mariadb_config = config.mariadb_config
        return f"mysql+aiomysql://{mariadb_config['user']}:{mariadb_config['password']}@{mariadb_config['host']}:{mariadb_config['port']}/{mariadb_config['database']}"
    
    @property
    def embedding_model(self) -> str:
        return config.embedding_config['model_id']
    
    @property
    def api_key(self) -> str:
        return "your_api_key"
    
    # .env에서 읽을 수 있는 추가 설정들 (선택적)
    huggingface_token: Optional[str] = None
    
    @property
    def server_host(self) -> str:
        return config.server_host
    
    @property
    def server_port(self) -> int:
        return config.server_port

    @model_validator(mode='after')
    def validate_model(self) -> 'Settings':
        if self.selected_model not in self.available_models:
            raise ValueError(f"Selected model '{self.selected_model}' is not one of the available models: {self.available_models}")
        return self

settings = Settings()
