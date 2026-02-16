import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Kitti3D
from pointnet import PointNetCls
import tensorstream_ops
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import numpy as np

def collate_fn(batch):
    points = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    points_padded = torch.nn.utils.rnn.pad_sequence(points, batch_first=True)
    points_padded = points_padded.transpose(2, 1)
    labels_stacked = torch.stack(labels)
    return points_padded, labels_stacked

def evaluate():
    # --- CONFIGURATION ---
    MODEL_PATH = "pointnet_epoch_10.pth" # Use your best epoch
    # Same data path as training
    DATA_DIR = os.path.expanduser('/scratch/kkota3/kaggle_cache/datasets/garymk/kitti-3d-object-detection-dataset/versions/1/training')
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Starting Evaluation on {DEVICE}...")

    # 1. Load Validation Data
    val_ds = Kitti3D(DATA_DIR, split='val')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, collate_fn=collate_fn)
    
    print(f"📂 Validation Samples: {len(val_ds)}")

    # 2. Load Model
    model = PointNetCls(k=2, feature_transform=True).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Loaded weights from {MODEL_PATH}")
    else:
        print(f"❌ Error: Model file {MODEL_PATH} not found.")
        return

    model.eval()

    # 3. Run Inference
    all_preds = []
    all_targets = []
    inference_times = []

    print("running inference...")
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Measure Inference Time (Pure Compute)
            start = time.perf_counter()
            pred, _, _ = model(data)
            end = time.perf_counter()
            inference_times.append((end - start) / data.size(0)) # Time per sample

            # Get predicted class (0 or 1)
            pred_choice = pred.data.max(1)[1]
            
            all_preds.extend(pred_choice.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 4. Calculate Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    avg_latency_ms = np.mean(inference_times) * 1000

    print("\n" + "="*40)
    print(f"📊 EVALUATION RESULTS")
    print("="*40)
    print(f"✅ Accuracy:       {accuracy*100:.2f}%")
    print(f"⚡ Avg Latency:    {avg_latency_ms:.4f} ms/sample")
    print(f"🏎️  Throughput:     {1000/avg_latency_ms:.0f} samples/sec")
    print("-" * 40)
    
    print("\n📝 Classification Report:")
    # Class 0 = Background, Class 1 = Car
    print(classification_report(all_targets, all_preds, target_names=["Background", "Car"]))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

if __name__ == "__main__":
    import os
    evaluate()