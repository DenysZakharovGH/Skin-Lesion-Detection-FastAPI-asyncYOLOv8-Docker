from glob import glob

import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

from train.f_RCNN_RESNET.config_RCNN_RESNET import dataset_image_dir, image_format, dataset_labels_dir, label_format, \
    val_image_dir, val_labels_dir

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SkinLesionDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __getitem__(self, idx):
        row = self.df[idx]

        self.transform = T.ToTensor()
        img = self.transform(Image.open(row['image_path']).convert("RGB"))

        boxes = row["boxes"].clone().detach().float()
        labels = row["labels"].clone().detach().long()

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.df)


def collate_fn(batch): return tuple(zip(*batch))


def get_train_dataloader():
    df = load_YOLO_dataset(dataset_image_dir, dataset_labels_dir)
    train_dataset = SkinLesionDataset(df, transforms=None)

    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=4,               # Скільки картинок модель бачить за раз
        shuffle=True,               # Перемішувати дані кожну епоху (обов'язково для тренування)
        num_workers=os.cpu_count(),              # Скільки ядер процесора готують дані (прискорює навчання)
        collate_fn=collate_fn       # Спеціальна функція для Faster R-CNN
    )
    return train_dataloader


def get_val_dataloader():
    df = load_YOLO_dataset(val_image_dir, val_labels_dir)
    train_dataset = SkinLesionDataset(df, transforms=None)

    return DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn
    )


def get_transforms():
    return A.Compose([
        A.Resize(600, 450),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc'))


def yolo_to_pytorch(yolo_line, img_w, img_h, separator):
    label, cx, cy, w, h = map(float, yolo_line.split(separator))

    xmin = (cx - w / 2) * img_w
    ymin = (cy - h / 2) * img_h
    xmax = (cx + w / 2) * img_w
    ymax = (cy + h / 2) * img_h

    boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
    labels = torch.tensor([int(label)], dtype=torch.int64)

    return labels, boxes


def load_YOLO_dataset(image_dir, labels_dir):
    dataFrame: list[dict] = []
    for image_path in glob(image_dir + "/*" + image_format):
        file_name = os.path.basename(image_path).replace(image_format, label_format)
        label_full_path = labels_dir + "/" + file_name
        if os.path.exists(label_full_path):
            img_h, img_w = cv2.imread(image_path).shape[:2]
            with open(label_full_path, "r") as f:
                labels, boxes = yolo_to_pytorch(f.read(), img_w, img_h, " ")
                dataFrame.append({"image_path": image_path, "labels": labels, "boxes": boxes})
    return dataFrame

def load_img(image_path:str=""):
    transform = T.ToTensor()
    img = transform(Image.open(image_path).convert("RGB"))
    return img








