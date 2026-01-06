from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_STORAGE = PROJECT_ROOT / "data_storage"
FRONTEND_STORAGE = PROJECT_ROOT / "frontend"

class RunConfig(BaseModel):
    host: str = "127.0.0.1"
    #host: str = "192.168.2.87"
    port: int = 8000

class LLMConfig(BaseModel):
    rate_limits: str = "5/minute"
    max_token_limits: int = 400
    treshold_answer_similarity: float = 0.5
    user_input_limits: int = 100
    temperature: int = 0.7

class Setting(BaseSettings):
    db_echo: bool = True  # settings variable to use for settings of our current DB
    run: RunConfig = RunConfig()
    llm: LLMConfig = LLMConfig()

settings = Setting()