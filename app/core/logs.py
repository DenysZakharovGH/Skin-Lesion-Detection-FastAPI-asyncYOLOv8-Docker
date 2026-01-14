import logging
import os

from app.core.app_config import LOGS_PATH

os.makedirs(LOGS_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),              # консоль
        logging.FileHandler(f"{LOGS_PATH}/app.log"),        # файл
    ],
)

logger = logging.getLogger("cnn_api")