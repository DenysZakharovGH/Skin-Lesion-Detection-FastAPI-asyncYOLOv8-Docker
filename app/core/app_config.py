import csv
from glob import glob
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_STORAGE = PROJECT_ROOT / "data_storage"
FRONTEND_STORAGE = PROJECT_ROOT / "frontend"
DESCRIPTION_PATH = PROJECT_ROOT / "lesion_description"
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
    "default": (255, 0, 255),
}


class FrontendConfig():

    lesion_desc_dict = {}
    def __init__(self, **data):
        super().__init__(**data)
        file = glob(f"{DESCRIPTION_PATH}/*.csv")
        if not file:
            raise ValueError(f"No file at {DESCRIPTION_PATH}/*.csv found")
        file_path = file[0]
        print(f"Found. {file_path} selected")
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.lesion_desc_dict = {row['class_code']: row for row in reader}
        if not self.lesion_desc_dict:
            raise ValueError(f"file at {DESCRIPTION_PATH}/*.csv is empty")

    def add_description(self, pred_class):
        return self.lesion_desc_dict[pred_class]


class RunConfig(BaseModel):
    host: str = "127.0.0.1"
    #host: str = "192.168.2.40"
    port: int = 8000
    rate_limits: str = "25/minute"

class CNNConfig(BaseModel):
    rate_limits: str = "5/minute"
    max_token_limits: int = 400
    user_input_limits: int = 100
    model_path_yolo: str = None
    model_path_resnet: str = None
    max_image_h_w: int = 4500

    iou: float = 0.5  # IoU threshold for NMS
    conf: float = 0.05

    def __init__(self, **data):
        super().__init__(**data)
        models = glob(f"{MODELS_PATH}/*yolo.pt")
        if not models:
            raise ValueError(f"No models at {MODELS_PATH}/*yolo.pt found")
        print(f"Found. {models[0]} selected")
        self.model_path_yolo = models[0]

        models = glob(f"{MODELS_PATH}/*resnet.pt")
        if not models:
            raise ValueError(f"No models at {MODELS_PATH}/*resnet.pt found")
        print(f"Found. {models[0]} selected")
        self.model_path_resnet = models[0]


class Setting(BaseSettings):
    db_echo: bool = True  # settings variable to use for settings of our current DB
    run: RunConfig = RunConfig()
    cnn: CNNConfig = CNNConfig()

settings = Setting()