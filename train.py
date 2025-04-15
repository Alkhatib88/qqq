#!/usr/bin/env python3
"""
train.py

Training script for the UnifiedMultimodalModel.
This script imports the model from model.py, creates a dummy dataset (configured with a comprehensive list
of training data sources for coding, game engines, reasoning, news, and more), and trains the model using an
optimized DataLoader to maximize GPU utilization and accelerate dataset loading.
It also demonstrates the model’s ability to create and execute scripts via its function call interface.
"""

import os, threading, math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import UnifiedMultimodalModel, DummyDataset, get_default_config

# Enable CuDNN benchmark for maximum GPU performance
torch.backends.cudnn.benchmark = True

def train_model(model, dataloader, num_epochs, learning_rate, device):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # For AMP mixed precision training
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Move batch data to device
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad()
            # Use AMP for mixed-precision training
            with autocast():
                outputs = model(batch)
                loss = 0.0
                # Compute text branch loss using cross-entropy
                if 'text_out' in outputs and 'text' in batch:
                    logits = outputs['text_out']
                    target = batch['text']
                    loss += criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                # Reconstruction losses for audio, image, and video
                if 'audio_out' in outputs:
                    loss += nn.MSELoss()(outputs['audio_out'], batch['audio'])
                if 'image_out' in outputs:
                    loss += nn.MSELoss()(outputs['image_out'], batch['image'])
                if 'video_out' in outputs:
                    loss += nn.MSELoss()(outputs['video_out'], batch['video'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "unified_model.pt")
    print("Training complete. Model saved as 'unified_model.pt'.")
    # Demonstrate function calls: build and execute a script
    build_result = model.call_function("build_script", "example_script.py")
    exec_result = model.call_function("execute_script", "example_script.py")
    print("Function Call Demo:")
    print(build_result)
    print(exec_result)
    # List the training datasets for reference
    print("\nTraining Datasets:")
    for ds in model.config.get("training_datasets", []):
        print(f" - {ds}")

def main():
    config = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedMultimodalModel(config).to(device)
    # Create dummy dataset
    dataset = DummyDataset(num_samples=50, config=config)
    # Use maximum number of worker threads and pin memory for fast prefetching
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=num_workers, pin_memory=True)
    num_epochs = 3
    learning_rate = 1e-4
    # Run training in a separate thread to allow asynchronous processing
    train_thread = threading.Thread(target=train_model, args=(model, dataloader, num_epochs, learning_rate, device))
    train_thread.start()
    train_thread.join()

if __name__ == "__main__":
    main()
