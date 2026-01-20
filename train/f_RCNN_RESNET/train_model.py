#Goal: Detect and classify skin lesions in dermatoscopic images from HAM10000.
# Dataset: HAM10000 (7 classes of lesions, ~10k images).
# Framework: PyTorch + torchvision + Lightning/Optuna for hyperparameter optimization.
# Extras:
# Automated hyperparameter tuning → Optuna
# Automated training pipelines → PyTorch Lightning
# Feature extraction → backbone outputs (ResNet)
# Monitoring → TensorBoard or Weights & Biases (wandb)
import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from train.f_RCNN_RESNET.config_RCNN_RESNET import num_classes
from train.f_RCNN_RESNET.dataset import get_train_dataloader, load_img, get_val_dataloader
from train.f_RCNN_RESNET.models import get_model, FasterRCNNLightning

# Step 1: Understanding the Problem

# Step 2: Prepare the Dataset
#df = load_YOLO_dataset()

# Step 3: Create a PyTorch Dataset
#train_dataset = SkinLesionDataset(df, transforms=None)

# # Step 4: Prepare the Model -> Transfer Learning
# base_model = get_model(num_classes)
#
# # Step 5: Training Loop (Lightning)
# model = FasterRCNNLightning(model=base_model, lr=0.005)

wandb_logger = WandbLogger(project="skin_lesion_detection")

checkpoint_cb = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="fasterrcnn-{epoch}-{val_loss:.3f}",
    dirpath="checkpoints"
)

wandb_logger.experiment.log({"message": f"hyperparameters search starts"})

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = get_model(num_classes=num_classes)
    lightning_model = FasterRCNNLightning(model, lr=lr)

    trainer = Trainer(
        max_epochs=1,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=3,
        callbacks = [checkpoint_cb],
    )

    trainer.fit(lightning_model,
                train_dataloaders = get_train_dataloader(),
                val_dataloaders = get_val_dataloader()
                )

    # Use validation loss as Optuna objective
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss


# Step 6: Hyperparameter Search (Optuna)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Best hyperparameters
best_params = study.best_params
best_trial_ckpt = checkpoint_cb.best_model_path
print("Best hyperparameters:", best_params)
print("Best best_trial_ckpt:", best_trial_ckpt)
wandb_logger.experiment.log({"message": f"Best hyperparameters: {best_params}"})
wandb_logger.experiment.log({"message": f"Best best_trial_ckpt: {best_trial_ckpt}"})

best_trial_ckpt = checkpoint_cb.best_model_path  # saved during Optuna trial
model = FasterRCNNLightning.load_from_checkpoint(best_trial_ckpt, model=get_model(num_classes))

trainer = Trainer(
    max_epochs=10,  # or more epochs for final training
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[checkpoint_cb],
    log_every_n_steps=1
)

# This is the final training run
trainer.fit(
    model,
    train_dataloaders=get_train_dataloader(),
    val_dataloaders=get_val_dataloader()
)

wandb_logger.experiment.log({"message": "Final model done"})





