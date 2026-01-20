import torch

from train.f_RCNN_RESNET.config_RCNN_RESNET import num_classes
from train.f_RCNN_RESNET.dataset import load_img
from train.f_RCNN_RESNET.models import FasterRCNNLightning, get_model

model = FasterRCNNLightning.load_from_checkpoint(
    "/home/denys/PycharmProjects/Classification-of-pigmented-skin-lesions/train/f_RCNN_RESNET/checkpoints/fasterrcnn-epoch=4-val_loss=0.139.ckpt",
    model=get_model(num_classes)
)


model.eval()


import cv2
import torch
import numpy as np

# Example class mapping for HAM10000
class_names = {
    1: "akiec",
    2: "bcc",
    3: "bkl",
    4: "df",
    5: "mel",
    6: "nv",
    7: "vasc"
}

def draw_predictions_opencv(img_tensor, predictions, class_names, threshold=0.3, save_path=None):
    # Convert tensor to BGR numpy
    from torchvision.transforms import ToPILImage
    import cv2, numpy as np

    img_cv = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    pred = predictions[0]
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']

    mask = scores > threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.int().cpu().numpy()
        cls_name = class_names[int(label)]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,0,255), 2)
        text = f"{cls_name}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_cv, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (0,0,255), -1)
        cv2.putText(img_cv, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imwrite("/home/denys/PycharmProjects/Classification-of-pigmented-skin-lesions/train/f_RCNN_RESNET/test_run.png", img_cv)

    return img_cv


img_tensor = load_img("/home/denys/PycharmProjects/Classification-of-pigmented-skin-lesions/train/f_RCNN_RESNET/test_image.jpg")  # tensor [C,H,W]
with torch.no_grad():
    predictions = model.model([img_tensor])
    output_img = draw_predictions_opencv(img_tensor, predictions, class_names, threshold=0.3)
    print(predictions)
