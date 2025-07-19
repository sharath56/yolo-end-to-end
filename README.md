# 🧠 YOLO Object Detection - End-to-End Guide (Basic to Advanced)

## 🔹 1. What is Object Detection?

Object detection is a computer vision technique that detects and classifies multiple objects in an image or video.

* **Task**: Identify *what* is in the image and *where* it is (bounding box).
* **Applications**:

  * Self-driving cars (detect pedestrians, lanes)
  * Surveillance (detect people)
  * Retail (track customers, detect products)
  * Drones, robotics, sports analytics

## 🔹 2. What is YOLO?

**YOLO (You Only Look Once)** is a family of real-time object detection models. Instead of scanning an image in parts, YOLO sees the image once and makes all predictions in a single forward pass.

### ✅ Key Benefits

* **Fast** (real-time)
* **Accurate** (optimized bounding boxes)
* **Versatile** (works with detection, segmentation, pose)

## 🔹 3. YOLO Tasks

| Task    | Output                             |
| ------- | ---------------------------------- |
| detect  | Bounding boxes + class label       |
| segment | Bounding boxes + mask + label      |
| pose    | Bounding boxes + keypoints + label |

## 🔹 4. YOLO Versions

| Version | Library        | Notes                                |
| ------- | -------------- | ------------------------------------ |
| YOLOv5  | Ultralytics    | PyTorch, easy to use                 |
| YOLOv8  | Ultralytics    | Better performance, supports segment |
| YOLOv11 | Community fork | Faster + supports advanced features  |

---

# 🚀 Getting Started with YOLO

## 1. 🧱 Dataset Structure

```text
data.yaml
images/
  train/
  val/
labels/
  train/
  val/
```

### `labels/*.txt` file format:

```
<class_id> <x_center> <y_center> <width> <height>
# All values normalized (0–1)
```

## 2. ⚙️ data.yaml Example

```yaml
path: ./dataset
train: images/train
val: images/val
nc: 3
names: ['person', 'car', 'bike']
```

---

# 🏋️‍♂️ Training a YOLO Model

### CLI Example:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

### 🔄 From Scratch:

```bash
yolo task=detect mode=train model=yolov8n.yaml data=data.yaml epochs=50 imgsz=640 pretrained=False
```

---

# 📊 Evaluating the Model

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

### Metrics:

* **mAP\@0.5**: Accuracy at IoU 50%
* **mAP\@0.5:0.95**: Strict accuracy over thresholds
* **Precision/Recall**

## ✅ How Evaluation Works Internally:

```python
# For each predicted bounding box:
# 1. Compute IoU (Intersection over Union) with all ground truth boxes
# 2. If IoU > threshold (e.g. 0.5), it's considered a correct prediction
# 3. If no matching ground truth box, it's a False Positive
# 4. Ground truths not matched are False Negatives
#
# Using TP, FP, FN:
# - Precision = TP / (TP + FP)
# - Recall = TP / (TP + FN)
# mAP is average precision computed across all classes
```

### 🔎 Modifying Output for Binary Classification:

If you're doing binary classification (e.g., object vs. background), you can:

* Replace last layer to output 1 class (instead of `num_classes`)
* Use sigmoid instead of softmax (for binary)
* Loss function changes from multi-class CE to binary CE (BCE)

---

# 🌟 Making Predictions

```bash
yolo task=detect mode=predict model=best.pt source=your_image.jpg
```

* For webcam: `source=0`

---

# 📈 Visualizing Training with TensorBoard

```bash
tensorboard --logdir runs/train
```

Open browser at `http://localhost:6006`

---

# 🔍 Advanced Concepts

## 🔹 Data Augmentation

* `mosaic`, `copy_paste`, `hsv_h/s/v`, `flipud`, `fliplr`

## 🔹 Hyperparameters

| Param         | Meaning               | Tip                     |
| ------------- | --------------------- | ----------------------- |
| lr0           | Initial learning rate | Lower for fine-tuning   |
| momentum      | SGD momentum          | Helps stabilize updates |
| weight\_decay | Regularization        | Prevents overfitting    |
| dropout       | Layer dropout         | Add for large models    |

## 🔹 Early Stopping

```yaml
patience: 10
```

Stops training early if no improvement.

---

# 🤖 Automating Evaluation & Plotting

### `evaluate_all_models.py`

```python
import os
import matplotlib.pyplot as plt
import pandas as pd

from ultralytics import YOLO

models_dir = "runs/segment/"
log_data = []

for folder in os.listdir(models_dir):
    model_path = os.path.join(models_dir, folder, "weights", "best.pt")
    if os.path.exists(model_path):
        model = YOLO(model_path)
        metrics = model.val()
        log_data.append({
            "model": folder,
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
        })

# Save CSV
df = pd.DataFrame(log_data)
df.to_csv("validation_scores.csv", index=False)

# Plot
plt.figure(figsize=(10,5))
plt.plot(df["model"], df["mAP50"], label="mAP@0.5", marker="o")
plt.xticks(rotation=45)
plt.grid(True)
plt.title("YOLO Model Comparison")
plt.xlabel("Model")
plt.ylabel("mAP@0.5")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_plot.png")
```

---

# 💡 Tips & Tricks

* Freeze backbone (`freeze: [0, 10]`) to train only head
* Use larger batch if GPU memory allows (improves gradient estimation)
* Use stratified validation (same class distribution in train/val)
* Check label noise or misalignments
* Use `conf=0.25` threshold for prediction filtering
* Try `auto_augment=randaugment` for stronger augmentation
* Lower `lr0` and increase `epochs` for fine-tuning
* Class imbalance? Try focal loss or weighted CE

---

# 📦 Exporting the Model

```bash
yolo mode=export format=onnx  # or torchscript, coreml
```

---

# 📘 Summary

| Step     | Purpose                               |
| -------- | ------------------------------------- |
| Dataset  | Provides inputs and labels            |
| Train    | Optimizes weights                     |
| Validate | Monitors performance on held-out data |
| Predict  | Applies model to unseen images        |
| Tune     | Improves performance systematically   |
| Deploy   | Use in apps, APIs, IoT, real-time     |

---

HAppieeee Codeing !!!

SHARATH VN !!!!!!!