import unittest

from PIL import Image


class TestVisionPipeline():
    """
    End-to-end tests for detection + classification pipeline.

    These tests verify:
    - Object is detected
    - Bounding box is approximately correct
    - Crop classification is correct and confident
    """
    def __init__(self):
        self.image = Image.open("tests/test_image.jpg").convert("RGB")

        # Detection expectations
        self.EXPECTED_BOX = [228.16, 155.75, 350.45, 281.53]
        self.MIN_IOU = 0.90  # tolerant to small model drift

        # Classification expectations
        self.EXPECTED_CLASS = "MEL"
        self.MIN_CONFIDENCE = 0.90

        self.expected_class_code = [
            "MEL",  # 1. Melanoma
            "NV",  # 2. Melanocytic nevi
            "BCC",  # 3. Basal cell carcinoma
            "AKIEC",  # 4. Actinic keratoses
            "BKL",  # 5. Benign keratosis-like lesions
            "DF",  # 6. Dermatofibroma
            "VASC",  # 7. Vascular lesions
        ]
        self.expected_keys = ["class_code",

                              "type_en",
                              "type_de",
                              "type_uk",

                              "danger_en",
                              "danger_de",
                              "danger_uk",

                              "advice_en",
                              "advice_de",
                              "advice_uk",

                              "description_en",
                              "description_de",
                              "description_uk",
                              ]


tests_constants = TestVisionPipeline()