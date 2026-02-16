import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Kitti3D
from pointnet import PointNetCls, feature_transform_regularizer
import os
import time

def collate_fn(batch):
    """
    Since C++ Voxelization returns different number of points per scan,
    we pad them to the same size so they fit in a GPU batch.
    """
    points = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad to the maximum number of points in this batch
    points_padded = torch.nn.utils.rnn.pad_sequence(points, batch_first=True)
    
    # Transpose for PointNet: (Batch, Points, 3) -> (Batch, 3, Points)
    points_padded = points_padded.transpose(2, 1)
    
    labels_stacked = torch.stack(labels)
    return points_padded, labels_stacked

def main():
    # 1. Configuration
    DATA_DIR = os.path.expanduser('/scratch/kkota3/kaggle_cache/datasets/garymk/kitti-3d-object-detection-dataset/versions/1/training')

    # 2. (Optional) Add a safety check to stop the script if it's still wrong
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Path does not exist: {DATA_DIR}")   
    BATCH_SIZE = 16 
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Starting TensorStream Training on {DEVICE}...")

    # 2. Setup Data Loaders
    train_ds = Kitti3D(DATA_DIR, split='train')
    val_ds = Kitti3D(DATA_DIR, split='val')
    
    # Num_workers=4 is safe because C++ releases the GIL
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn)
    
    print(f"Loaded {len(train_ds)} training samples.")

    # 3. Initialize Model (k=2 for Car vs No-Car)
    model = PointNetCls(k=2, feature_transform=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss() # PointNet uses log_softmax

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE).long()
            
            optimizer.zero_grad()
            
            # Forward Pass
            pred, trans, trans_feat = model(data)
            
            # Loss
            loss = criterion(pred, target)
            if trans_feat is not None:
                loss += feature_transform_regularizer(trans_feat) * 0.001
                
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct += pred_choice.eq(target.data).cpu().sum()
            total_samples += target.size(0)
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}: Batch {batch_idx} | Loss: {loss.item():.4f}")

        # Epoch Summary
        epoch_acc = 100. * correct / total_samples
        print(f"✅ Epoch {epoch+1} Completed in {time.time() - start_time:.2f}s")
        print(f"   Avg Loss: {total_loss/len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"pointnet_epoch_{epoch+1}.pth")

    print("🎉 Training Complete. Model saved.")

if __name__ == "__main__":
    main()