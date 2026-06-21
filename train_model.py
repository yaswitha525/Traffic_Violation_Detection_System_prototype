import os
import shutil

from ultralytics import YOLO


DATA_YAML = os.path.join("master_traffic_violation_dataset", "data.yaml")
DEFAULT_RUN_NAME = "traffic_violation_yolov8"
OUTPUT_MODEL = "best_traffic_model.pt"


def main():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    model = YOLO("yolov8s.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project="runs/detect",
        name=DEFAULT_RUN_NAME,
        exist_ok=True,
    )

    best_pt = os.path.join("runs", "detect", DEFAULT_RUN_NAME, "weights", "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"Training completed but best weights were not found at: {best_pt}")

    shutil.copy2(best_pt, OUTPUT_MODEL)
    print(f"Training complete. Best weights copied to {OUTPUT_MODEL}")
    print(f"Ultralytics results object: {results}")


if __name__ == "__main__":
    main()