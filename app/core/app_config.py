from glob import glob
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_STORAGE = PROJECT_ROOT / "data_storage"
FRONTEND_STORAGE = PROJECT_ROOT / "frontend"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"

COLORS = {
    "AKIEC": (255, 0, 0),
    "BCC": (0, 255, 0),
    "BKL": (0, 0, 255),
    "DF": (255, 255, 0),
    "MEL": (255, 0, 255),  # melanoma → magenta
    "NV": (0, 255, 255),
    "VASC": (128, 128, 255),
}

class RunConfig(BaseModel):
    host: str = "127.0.0.1"
    #host: str = "192.168.2.87"
    port: int = 8000
    rate_limits: str = "25/minute"

class CNNConfig(BaseModel):
    rate_limits: str = "500/minute"
    max_token_limits: int = 400
    user_input_limits: int = 100
    model_path: str = None

    def __init__(self, **data):
        super().__init__(**data)
        models = glob(f"{MODELS_PATH}/*.pt")
        if not models:
            raise ValueError(f"No models at {MODELS_PATH} found")
        self.model_path = models[0]


class Setting(BaseSettings):
    db_echo: bool = True  # settings variable to use for settings of our current DB
    run: RunConfig = RunConfig()
    cnn: CNNConfig = CNNConfig()

settings = Setting()