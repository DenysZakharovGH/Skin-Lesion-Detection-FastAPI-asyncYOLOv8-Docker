import unittest
from glob import glob

from app.core.app_config import DESCRIPTION_PATH, FrontendConfig, MODELS_PATH
from tests.constants import tests_constants


class MyTestCase(unittest.TestCase):

    def test_description_file(self):
        file = glob(f"{DESCRIPTION_PATH}/*.csv")
        assert bool(file), f"File {DESCRIPTION_PATH}/*.csv not exists"

    def test_description_content(self):
        config = FrontendConfig()
        for _class_code_ in tests_constants.expected_class_code:
            info_return = config.add_description(_class_code_)
            difference = list(set(tests_constants.expected_keys) - set(info_return.keys()))
            if difference:
                raise ValueError(f"File {DESCRIPTION_PATH}/*.csv does not contain {difference} keys")
        assert True

    def test_yolo_file(self):
        models = glob(f"{MODELS_PATH}/*yolo.pt")
        assert bool(models), f"No models at {MODELS_PATH}/*yolo.pt found"

    def test_resnet_file(self):
        models = glob(f"{MODELS_PATH}/*resnet.pt")
        assert bool(models), f"No models at {MODELS_PATH}/*resnet.pt found"



if __name__ == '__main__':
    unittest.main()
