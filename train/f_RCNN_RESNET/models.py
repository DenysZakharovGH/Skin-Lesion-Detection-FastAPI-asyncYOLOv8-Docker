
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights



import pytorch_lightning as pl
import torch


def get_model(num_classes):
    # Load pre-trained Faster R-CNN

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # -> Head Replacement
    return model


class FasterRCNNLightning(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(img.to(self.device) for img in images)  # .to(self.device) send info to GPU
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]  # .to(self.device) send info to GPU

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # 🔑 Force loss computation
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
            val_loss = sum(loss for loss in loss_dict.values())

        self.model.eval()

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer