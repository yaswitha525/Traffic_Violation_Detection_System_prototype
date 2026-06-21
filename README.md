# Traffic Violation Detection

## Dataset

The dataset is stored in `master_traffic_violation_dataset` with this structure:

- `data.yaml`
- `train/images`, `train/labels`
- `valid/images`, `valid/labels`
- `test/images`, `test/labels`

## Train the model

Run:

```bash
python train_model.py
```

This will train YOLOv8 from `master_traffic_violation_dataset/data.yaml` and copy the best weights to `best_traffic_model.pt` in the project root.

## Run the app

```bash
streamlit run app.py
```

If `best_traffic_model.pt` is not present yet, you can upload the trained `.pt` file from the app sidebar.