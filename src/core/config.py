from pydantic_settings import BaseSettings
from pydantic import model_validator, ConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8',
        extra='ignore'  # .env의 추가 필드 무시
    )
    
    # 사용자가 선택할 수 있는 모델 목록 (3개로 제한)
    available_models: List[str] = ["qwen2.5-7b", "llama3.1-8b", "gemma-3-4b"]
    
    # 사용자가 선택한 모델
    selected_model: str = "qwen2.5-7b"
    
    # 선택적 설정들
    database_url: str = "sqlite:///./database.db"
    embedding_model: str = "BAAI/bge-m3"
    api_key: str = "your_api_key"
    
    # .env에서 읽을 수 있는 추가 설정들 (선택적)
    huggingface_token: Optional[str] = None
    server_host: str = "0.0.0.0"
    server_port: int = 8001

    @model_validator(mode='after')
    def validate_model(self) -> 'Settings':
        if self.selected_model not in self.available_models:
            raise ValueError(f"Selected model '{self.selected_model}' is not one of the available models: {self.available_models}")
        return self

settings = Settings()
