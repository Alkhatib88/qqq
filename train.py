#!/usr/bin/env python3
"""
train.py

Training script for the UnifiedMultimodalModel.
This script imports the model from model.py, creates a dummy dataset configured
with a comprehensive list of training data sources (covering coding, game engines, 
3D tools, reasoning, history, news, etc.), and trains the model using an optimized DataLoader 
that maximizes GPU utilization via AMP and multi-threading.
Progress information is displayed via tqdm.
"""

import os, threading
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: simply return the iterable (no dynamic progress display)
    def tqdm(iterable, **kwargs):
        return iterable

from model import UnifiedMultimodalModel, DummyDataset, get_default_config

# Enable CuDNN benchmark for maximum GPU performance
torch.backends.cudnn.benchmark = True

def train_model(model, dataloader, num_epochs, learning_rate, device):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device=device)  # For AMP mixed precision training
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_counter = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in progress_bar:
            # Move each tensor to the appropriate device
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(batch)
                loss = 0.0
                # Text branch loss (cross-entropy)
                if "text_out" in outputs and "text" in batch:
                    logits = outputs["text_out"]  # shape (B, seq_len, vocab_size)
                    target = batch["text"]
                    loss += criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                # Reconstruction losses for audio, image, and video
                if "audio_out" in outputs:
                    loss += nn.MSELoss()(outputs["audio_out"], batch["audio"])
                if "image_out" in outputs:
                    loss += nn.MSELoss()(outputs["image_out"], batch["image"])
                if "video_out" in outputs:
                    loss += nn.MSELoss()(outputs["video_out"], batch["video"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batch_counter += 1
            if hasattr(progress_bar, "set_postfix"):
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = epoch_loss / batch_counter
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "unified_model.pt")
    print("Training complete. Model saved as 'unified_model.pt'.")
    
    # Demonstrate function calls: build and execute a script.
    build_result = model.call_function("build_script", "example_script.py")
    execute_result = model.call_function("execute_script", "example_script.py")
    print("Function Call Demo:")
    print(build_result)
    print(execute_result)
    
    # Display the list of training datasets.
    print("\nTraining Datasets:")
    for ds in model.config.get("training_datasets", []):
        print(f" - {ds}")

def main():
    config = get_default_config()
    # GPU auto-detection and management:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        gpu_props = torch.cuda.get_device_properties(device)
        total_memory_gb = gpu_props.total_memory / (1024 ** 3)
        print(f"Using GPU: {gpu_name} - Total Memory: {total_memory_gb:.2f} GB")
        # Set GPU process memory fraction to 95%
        torch.cuda.set_per_process_memory_fraction(0.95, device)
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")
    
    model = UnifiedMultimodalModel(config).to(device)
    dataset = DummyDataset(num_samples=50, config=config)
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=num_workers, pin_memory=True)
    num_epochs = 3
    learning_rate = 1e-4
    
    # Run training in a separate thread to allow asynchronous processing.
    train_thread = threading.Thread(target=train_model, args=(model, dataloader, num_epochs, learning_rate, device))
    train_thread.start()
    train_thread.join()

if __name__ == "__main__":
    main()
