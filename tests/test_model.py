import unittest
from app.model import model_inference_sync, _classify_crop
from tests.constants import tests_constants


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - inter_area
    return inter_area / union if union else 0


def assert_iou_above(box, expected_box, min_iou):
    score = iou(box, expected_box)
    assert score >= min_iou, f"IoU too low: {score:.3f}"



class MyTestCase(unittest.TestCase):

    def test_yolo_detects_object_with_correct_location(self):
        results = model_inference_sync(tests_constants.image)
        self.assertTrue(results, "No detections returned")

        boxes = results[0].boxes.xyxy.cpu().numpy()
        self.assertGreater(len(boxes), 0, "YOLO returned zero boxes")

        best_box = boxes[0]  # assume top-1 detection
        assert_iou_above(best_box, tests_constants.EXPECTED_BOX, tests_constants.MIN_IOU)

    def test_classifier_predicts_correct_class_with_confidence(self):
        image_local = tests_constants.image.copy()
        results = model_inference_sync(image_local)
        self.assertTrue(results, "No detections returned")

        boxes = results[0].boxes.xyxy.cpu().numpy()
        self.assertGreater(len(boxes), 0, "No boxes to classify")

        r = results[0]

        classifications = []
        for bbox in results[0].boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox.cpu().numpy()
            crop = image_local.crop((x1, y1, x2, y2))
            classifications.append(_classify_crop(crop))

        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for _, (pred_class, prob), _, _ in zip(boxes, classifications, scores, classes):

                self.assertEqual(
                    pred_class,
                    tests_constants.EXPECTED_CLASS,
                    f"Wrong class predicted: {pred_class}",
                )

                self.assertGreaterEqual(
                    float(prob),
                    tests_constants.MIN_CONFIDENCE,
                    f"Confidence too low: {prob}",
                )


if __name__ == '__main__':
    unittest.main()
