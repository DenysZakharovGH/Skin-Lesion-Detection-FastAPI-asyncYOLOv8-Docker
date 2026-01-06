from ultralytics import YOLO

from config import PROJECT_ROOT

folder_path = f"{PROJECT_ROOT}/runs/detect/train9/weights"

dataset_image_dir = f"{PROJECT_ROOT}" + "/train/dataset/images"

model_paths = [f"{folder_path}/best.pt",
               f"{folder_path}/epoch90.pt",
               f"{folder_path}/last.pt"]
test_images = dataset_image_dir + "/val/"
data_yaml = f"{PROJECT_ROOT}" + "/train/dataset/custom_data.yaml"

for mp in model_paths:
    model = YOLO(mp)


    # Run validation
    metrics = model.val(data=data_yaml, imgsz=640)  # returns DetMetrics


    print(dir(metrics.box))  # list all attributes in the 'box' metric

    break

    print(f"Model: {mp}")

    # Overall box mAP metrics
    print("Overall mAP metrics:")
    print(f"  mAP50: {metrics.box.map50:.3f}")


    # Per-class mAP50
    print("Per-class mAP50:")
    for i, cls_name in enumerate(metrics.names):
        cls_map50 = metrics.box.map_per_class[i]  # float
        print(f"  {cls_name}: {cls_map50:.3f}")

    print("--------------------------")