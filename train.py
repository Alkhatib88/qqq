#!/usr/bin/env python3
"""
train.py

Training script for TitanModel on a RunPod H200 GPU (or CPU fallback).

Loads multiple Hugging Face datasets:
  - Multimodal-Fatima/VQAv2_sample_train
  - Multimodal-Fatima/OxfordFlowers_test
  - matlok/multimodal-python-copilot-training-overview
  - notbadai/python_functions_reasoning
  - espejelomar/code_search_net_python_10000_examples
  - reshinthadith/synthetic_program_synthesis_python_1M
  - suriyagunasekar/stackoverflow-python-with-meta-data
  - Sridevi/python_textbooks
  - nuprl/stack-dedup-python-testgen-starcoder-filter-v2

It builds a vocabulary, preprocesses samples, and trains TitanModel.
Shows average loss, sample token accuracy, shapes of outputs, and a progress bar for each epoch.

Training stops when sample token accuracy reaches 100% or the max 
number of epochs is reached.

Modifications made:
  • DataLoader uses num_workers=8 and pin_memory=True.
  • Automatic Mixed Precision (AMP) training is enabled using torch.cuda.amp.autocast and torch.cuda.amp.GradScaler.
  • cuDNN benchmark is enabled for optimized GPU performance.
  • Non-blocking transfers help speed up data movement from CPU to GPU.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from model import TitanModel
from PIL import Image
import json
from collections import Counter
import traceback
from tqdm import tqdm  # progress bar

BATCH_SIZE = 4
VOCAB_SIZE = 5000

def build_vocab(dataset, text_field="text", max_vocab_size=VOCAB_SIZE):
    counter = Counter()
    for sample in dataset:
        text = sample.get(text_field, "")
        tokens = text.split()
        counter.update(tokens)
    vocab_list = [word for word, _ in counter.most_common(max_vocab_size - 2)]
    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, word in enumerate(vocab_list, start=2):
        vocab[word] = idx
    return vocab

def tokenize_text(text, vocab, max_length=128):
    tokens = text.split()
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens][:max_length]
    if len(token_ids) < max_length:
        token_ids += [vocab["<pad>"]] * (max_length - len(token_ids))
    return token_ids

def decode_tokens(token_ids, rev_vocab):
    tokens = [rev_vocab.get(str(tok), "<unk>") for tok in token_ids if tok != 0]
    return " ".join(tokens)

def preprocess_sample(sample):
    out = {}
    text_keys = ["question", "caption", "text", "code"]
    for key in text_keys:
        if key in sample and sample[key] is not None:
            out["text"] = sample[key]
            break
    if "text" not in out:
        out["text"] = ""
    if "answer" in sample and sample["answer"]:
        out["target"] = sample["answer"]
    else:
        out["target"] = out["text"]
    if "image" in sample and sample["image"] is not None:
        out["image"] = sample["image"]
    else:
        out["image"] = None
    return out

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def collate_fn(batch, vocab):
    texts = []
    targets = []
    images = []
    for sample in batch:
        texts.append(tokenize_text(sample["text"], vocab))
        targets.append(tokenize_text(sample["target"], vocab))
        if sample["image"] is not None:
            if not isinstance(sample["image"], Image.Image):
                try:
                    sample["image"] = Image.fromarray(sample["image"])
                except Exception:
                    sample["image"] = Image.new("RGB", (64, 64))
            images.append(image_transform(sample["image"]))
        else:
            images.append(torch.zeros(3, 64, 64))
    text_tensor = torch.tensor(texts, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    image_tensor = torch.stack(images)
    batch_size = text_tensor.size(0)
    audio_tensor = torch.randn(batch_size, 1, 16000)
    video_tensor = torch.randn(batch_size, 3, 16, 64, 64)
    return {
        "text": text_tensor,
        "target": target_tensor,
        "image": image_tensor,
        "audio": audio_tensor,
        "video": video_tensor
    }

def load_and_prepare_datasets():
    dataset_ids = [
        "Multimodal-Fatima/VQAv2_sample_train",
        "Multimodal-Fatima/OxfordFlowers_test",
        "matlok/multimodal-python-copilot-training-overview",
        "notbadai/python_functions_reasoning",
        "espejelomar/code_search_net_python_10000_examples",
        "reshinthadith/synthetic_program_synthesis_python_1M",
        "suriyagunasekar/stackoverflow-python-with-meta-data",
        "Sridevi/python_textbooks",
        "nuprl/stack-dedup-python-testgen-starcoder-filter-v2"
    ]
    processed_datasets = []
    for ds_id in dataset_ids:
        try:
            split = "train" if "OxfordFlowers" not in ds_id else "test"
            ds = load_dataset(ds_id, split=f"{split}[:1%]")
            ds = ds.map(preprocess_sample)
            processed_datasets.append(ds)
            print(f"Loaded and processed dataset: {ds_id}")
        except Exception as e:
            print(f"Error loading {ds_id}: {e}")
    if processed_datasets:
        combined = concatenate_datasets(processed_datasets)
        combined = combined.shuffle(seed=42)
        return combined
    else:
        raise ValueError("No datasets loaded successfully.")

def main():
    try:
        # Enable cuDNN benchmark for optimized GPU performance
        torch.backends.cudnn.benchmark = True

        # 1. Load and prepare dataset
        dataset = load_and_prepare_datasets()

        # 2. Build vocabulary and save to file
        vocab = build_vocab(dataset, text_field="text", max_vocab_size=VOCAB_SIZE)
        with open("vocab.json", "w") as f:
            json.dump(vocab, f)
        print("Vocabulary built and saved to vocab.json.")
        
        # 3. Create reverse vocabulary mapping
        rev_vocab = {str(idx): word for word, idx in vocab.items()}

        # 4. Create a DataLoader with increased workers and pinned memory
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, vocab),
            num_workers=8,
            pin_memory=True
        )

        # 5. Define model config (must match training config)
        config = {
            "text_vocab_size": VOCAB_SIZE,
            "text_embed_dim": 512,
            "text_encoder_layers": 2,
            "text_decoder_layers": 2,
            "text_num_heads": 8,
            "text_ff_dim": 1024,
            "text_max_len": 128,
            "image_latent_dim": 256,
            "audio_latent_dim": 256,
            "audio_output_length": 16000,
            "video_latent_dim": 256,
            "video_output_shape": (3, 16, 64, 64),
            "fused_dim": 512,
            "attention_num_heads": 8,
            "attention_latent_dim": 64,
            "cot_decoder_layers": 2,
            "cot_max_len": 256,
            "rag_documents": [
                "Doc1: Multimodal dataset sample.",
                "Doc2: Python code reasoning examples.",
                "Doc3: Advanced topics in computer vision and language."
            ]
        }

        # 6. Setup device and load model fully on GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TitanModel(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Set GPU memory fraction (reserve 10% headroom)
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.9, device=device)
            torch.cuda.empty_cache()
            print("GPU memory limited to 90% of total (10% reserved).")

        # Setup Automatic Mixed Precision (AMP) using the old GradScaler API
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        # 7. Training loop
        model.train()
        num_epochs = 100
        epoch = 0
        while epoch < num_epochs:
            epoch += 1
            running_loss = 0.0
            batch_count = 0

            for batch in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} Progress", leave=False):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(
                        text_input_ids=batch["text"].to(device, non_blocking=True),
                        image_input=batch["image"].to(device, non_blocking=True),
                        audio_input=batch["audio"].to(device, non_blocking=True),
                        video_input=batch["video"].to(device, non_blocking=True),
                        query_ids=batch["target"].to(device, non_blocking=True)
                    )
                    loss = F.cross_entropy(
                        outputs["text_output_logits"].view(-1, config["text_vocab_size"]),
                        batch["target"].to(device, non_blocking=True).view(-1)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / batch_count
            print(f"\nEpoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")

            # 8. Quick validation check
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(loader))
                sample_text = sample_batch["text"].to(device, non_blocking=True)
                sample_target = sample_batch["target"].to(device, non_blocking=True)
                sample_outputs = model(
                    text_input_ids=sample_text,
                    image_input=sample_batch["image"].to(device, non_blocking=True),
                    audio_input=sample_batch["audio"].to(device, non_blocking=True),
                    video_input=sample_batch["video"].to(device, non_blocking=True),
                    query_ids=sample_target
                )
                predicted_tokens = torch.argmax(sample_outputs["text_output_logits"], dim=-1)
                input_decoded = decode_tokens(sample_text[0].cpu().tolist(), rev_vocab)
                target_decoded = decode_tokens(sample_target[0].cpu().tolist(), rev_vocab)
                predicted_decoded = decode_tokens(predicted_tokens[0].cpu().tolist(), rev_vocab)
                print("Sample Input Text:  ", input_decoded)
                print("Sample Target Text: ", target_decoded)
                print("Sample Predicted:   ", predicted_decoded)
                
                correct = (predicted_tokens[0] == sample_target[0]).sum().item()
                total = sample_target[0].numel()
                acc = correct / total * 100.0
                print(f"Sample Token Accuracy: {acc:.2f}%")
            model.train()

            print(f"Epoch {epoch} completed ({(epoch / num_epochs) * 100:.1f}% done)")
            if acc >= 100.0:
                print("Test passed 100%! Stopping training.")
                break

        print("\nTraining completed.")

    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
