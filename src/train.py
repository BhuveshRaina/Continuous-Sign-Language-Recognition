import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import config
from dataset import CSLDataset
from model import CSLModel

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on device: {device} (Max Accuracy Mode)")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(config.MODELS_DIR, 'best_csl_model_v2.pth')
    checkpoint_path = os.path.join(config.MODELS_DIR, 'last_checkpoint.pth')

    train_dataset = CSLDataset(split='train')
    val_dataset = CSLDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    num_classes = len(train_dataset.label_map)

    model = CSLModel(input_dim=config.FEATURE_DIM, num_classes=num_classes).to(device)
    
    # --- OPTIMIZED FOR ACCURACY ---
    # Label Smoothing penalizes overconfidence, great for small datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    # AdamW includes better Weight Decay (L2 regularization) than standard Adam
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Cosine Annealing slowly brings the learning rate down in a smooth curve
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    scaler = GradScaler('cuda')
    
    start_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint. Resuming...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"▶️ Resuming from Epoch {start_epoch+1}, Best Val Acc: {best_val_acc:.2f}%")

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        for frames, labels in train_loop:
            frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(frames)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(frames)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = (val_correct / val_total) * 100
        scheduler.step() # Step Cosine Annealing

        print(f"Epoch {epoch+1:02d}/{config.EPOCHS} | Train Acc: {(train_correct/train_total)*100:.2f}% | Val Acc: {epoch_val_acc:.2f}%")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_val_acc
        }, checkpoint_path)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ⭐ New Best Accuracy! Saved to {best_model_path}")

    print(f"\n🎉 Training Complete! Highest Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()