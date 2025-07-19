import os
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

# === Suggestions for Improvement ===
# 1. If mAP is low:
#    - Clean label noise
#    - Use stronger model (YOLOv11l or v11x)
#    - Increase input size (imgsz > 640)
#    - Tune augmentations: hsv, mosaic, copy_paste
#
# 2. If overfitting (train loss ↓ val mAP ↓):
#    - Use dropout
#    - Increase augmentation strength
#    - Reduce training epochs
#    - Freeze early layers and fine-tune later
#
# 3. If underfitting (all metrics low):
#    - Train longer
#    - Use better optimizer/lr scheduler (cos_lr)
#    - Use pretrained=True and avoid freezing
#
# 4. If class imbalance:
#    - Compute and apply class weights
#    - Use `single_cls=True` if only one general object class matters
#
# 5. Always:
#    - Monitor training in TensorBoard
#    - Save model every N epochs (save_period)
#    - Compare `best.pt`, `last.pt`, and mid-checkpoints



# === Configuration ===
weights_dir = "runs/segment/50_epoch_model_2"  # Folder containing .pt model files
data_yaml = "data.yaml"                        # Path to YOLO dataset config
test_image = "test/sample.jpg"                 # Image to test model predictions
output_dir = "model_eval_outputs"              # Where to store results
plot_dir = os.path.join(output_dir, "plots")

# === Setup ===
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# === Collect and sort model checkpoints ===
models = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pt")])
all_results = []

# === Evaluate all models ===
for model_file in models:
    model_path = os.path.join(weights_dir, model_file)
    epoch_label = os.path.splitext(model_file)[0]
    print(f"\n🔎 Evaluating {epoch_label}...")

    # Load model checkpoint
    model = YOLO(model_path)

    # === YOLO Validation ===
    # model.val() runs the validation loop:
    # - Loads validation dataset from data_yaml
    # - Runs inference on each image
    # - Computes mAP@0.5, mAP@0.5:0.95, losses (box, class, segmentation)
    # - Can optionally save confusion matrix, prediction plots, etc.
    metrics = model.val(data=data_yaml, split="val", plots=False, save_json=False)

    # === Predict a sample image to visually inspect results ===
    prediction = model.predict(
        source=test_image,
        conf=0.3,
        save=True,
        project=output_dir,
        name=epoch_label,
        save_txt=False,
        save_crop=False
    )

    # === Log validation results ===
    all_results.append({
        "model": epoch_label,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Box Loss": metrics.box.loss,
        "Seg Loss": metrics.seg.loss if hasattr(metrics, 'seg') else 0,
        "Class Loss": metrics.cls.loss
    })

# === Save results as CSV ===
df = pd.DataFrame(all_results)
csv_path = os.path.join(output_dir, "model_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\n📄 Saved CSV to {csv_path}")

# === Plot mAP over epochs ===
plt.figure(figsize=(12, 6))
plt.plot(df["model"], df["mAP50"], label="mAP@0.5", marker='o')
plt.plot(df["model"], df["mAP50-95"], label="mAP@0.5:0.95", marker='x')
plt.title("Model Performance (mAP)")
plt.xlabel("Model Checkpoint")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "mAP_scores.png"))

# === Plot all loss types ===
plt.figure(figsize=(12, 6))
plt.plot(df["model"], df["Box Loss"], label="Box Loss", marker='o')
plt.plot(df["model"], df["Seg Loss"], label="Seg Loss", marker='x')
plt.plot(df["model"], df["Class Loss"], label="Class Loss", marker='^')
plt.title("Model Losses")
plt.xlabel("Model Checkpoint")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "loss_plot.png"))

print(f"\n📊 Plots saved to {plot_dir}")

