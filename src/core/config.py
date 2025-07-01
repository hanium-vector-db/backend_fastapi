from pydantic import BaseSettings

class Settings(BaseSettings):
    model_path: str = "path/to/your/model"
    database_url: str = "sqlite:///./database.db"
    embedding_model: str = "BAAI/bge-m3"
    api_key: str = "your_api_key"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()