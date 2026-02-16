from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import time
import torch
import tensorstream_ops  # Your C++ Extension
from pointnet import PointNetCls  # Your Model Class
import uvicorn

# --- CONFIGURATION ---
MODEL_PATH = "pointnet_epoch_10.pth" # Ensure you have downloaded this file!

# FORCE CPU
DEVICE = torch.device("cpu")
print(f"🐢 Running strictly on: {DEVICE}")

app = FastAPI(title="TensorStream Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ---
print(f"🚀 Loading Model...")
model = PointNetCls(k=2, feature_transform=True).to(DEVICE)

if os.path.exists(MODEL_PATH):
    # map_location='cpu' is CRITICAL when moving from GPU training -> CPU inference
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"✅ Loaded weights from {MODEL_PATH}")
else:
    print(f"⚠️  WARNING: {MODEL_PATH} not found! Using random weights.")

model.eval()

# --- SERVE FRONTEND ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("frontend/index.html"):
        with open("frontend/index.html", "r") as f:
            return f.read()
    return "<h1>Error: index.html not found</h1>"

# --- INFERENCE ---
@app.post("/process_lidar")
async def process_lidar(file: UploadFile = File(...)):
    temp_path = f"temp_uploads/{file.filename}"
    os.makedirs("temp_uploads", exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        start_time = time.perf_counter()
        
        # 1. C++ Extension (CPU Optimized)
        points = tensorstream_ops.load_kitti_bin(temp_path)
        
        # 2. Voxel Downsample
        xyz = points[:, :3]
        processed = tensorstream_ops.voxel_downsample(xyz, 0.5)
        
        # 3. Model Inference (On CPU)
        # Unsqueeze adds batch dim: (N, 3) -> (1, N, 3)
        # Transpose fits PointNet: (1, N, 3) -> (1, 3, N)
        input_tensor = processed.unsqueeze(0).transpose(2, 1).to(DEVICE)
        
        with torch.no_grad():
            pred, _, _ = model(input_tensor)
            probs = torch.exp(pred)
            cls = torch.argmax(probs, dim=1).item()
            confidence = probs[0][cls].item() * 100

        end_time = time.perf_counter()
        
        label_map = {0: "BACKGROUND", 1: "CAR DETECTED"}
        result_label = label_map.get(cls, "UNKNOWN")
        
        os.remove(temp_path)
        
        return {
            "filename": file.filename,
            "original_points": points.shape[0],
            "processed_points": processed.shape[0],
            "processing_latency_ms": f"{(end_time - start_time) * 1000:.2f} ms",
            "prediction": result_label,
            "confidence": f"{confidence:.1f}%",
            "backend": "C++ Extension (CPU Mode)"
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)