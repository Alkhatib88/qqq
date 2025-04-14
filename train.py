#!/usr/bin/env python3
"""
train.py

Training script for TitanModel on a RunPod H200 GPU (or CPU fallback).

Loads multiple Hugging Face datasets including:
  • Multimodal-Fatima/VQAv2_sample_train
  • Multimodal-Fatima/OxfordFlowers_test
  • notbadai/python_functions_reasoning
  • espejelomar/code_search_net_python_10000_examples
  • reshinthadith/synthetic_program_synthesis_python_1M
  • suriyagunasekar/stackoverflow-python-with-meta-data
  • Sridevi/python_textbooks
  • nuprl/stack-dedup-python-testgen-starcoder-filter-v2
—and many reasoning/chain-of-thought datasets:
  • nvidia/OpenCodeReasoning
  • nvidia/Llama-Nemotron-Post-Training-Dataset
  • open-thoughts/OpenThoughts2-1M
  • glaiveai/reasoning-v1-20m
  • emilbiju/Execution-Dagger-Data-Math-think
  • wikimedia/wikipedia
  • FreedomIntelligence/medical-o1-reasoning-SFT
  • facebook/natural_reasoning
  • KingNish/reasoning-base-20k
  • ProlificAI/social-reasoning-rlhf
  • dvilasuero/natural-science-reasoning
  • smirki/UI_Reasoning_Dataset
  • reasoning-machines/gsm-hard
  • di-zhang-fdu/R1-Vision-Reasoning-Instructions
  • lightblue/reasoning-multilingual-R1-Llama-70B-train
  • prithivMLmods/Deepthink-Reasoning
  • Nan-Do/SPP_30K_reasoning_tasks
  • davanstrien/reasoning-required
  • antiven0m/physical-reasoning-dpo
  • isaiahbjork/cot-logic-reasoning
  • efficientscaling/Z1-Code-Reasoning-107K
  • iamtarun/python_code_instructions_18k_alpaca
  • flytech/python-codes-25k
  • Vezora/Tested-143k-Python-Alpaca
  • matlok/multimodal-python-copilot-training-overview
  • semeru/code-text-python
  • microsoft/LCC_python
  • thomwolf/github-python
  • Jofthomas/hermes-function-calling-thinking-V1
  • UCSC-VLAA/VLAA-Thinking
  • minchyeom/thinker-formatted
  • fhai50032/GPQA-Thinking-O1
  • ThinkAgents/Function-Calling-with-Chain-of-Thoughts
  • Salesforce/xlam-function-calling-60k

This script builds a vocabulary, preprocesses samples, and trains TitanModel.
It monitors GPU utilization via nvidia-smi and adjusts the DataLoader batch size
each epoch to try to “throttle” GPU usage toward a target level (e.g. 95%).

Additional improvements:
  • DataLoader uses num_workers=8 and pin_memory=True.
  • Automatic Mixed Precision (AMP) training is enabled.
  • cuDNN benchmark is enabled for optimized GPU performance.
  • Non-blocking data transfers and per-process GPU memory settings are used.
  • Out-of-memory errors are caught and the batch size is reduced if needed.
  • Extra reasoning fields (e.g. thinking, task, function call) are appended to training text.
"""

import math
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
from tqdm import tqdm
import torch.nn as nn
import subprocess
import sys
import time

# Constants for vocabulary/building and GPU throttling
INITIAL_BATCH_SIZE = 4
VOCAB_SIZE = 5000
TARGET_GPU_UTILIZATION = 95   # Target GPU utilization (percent)
MAX_BATCH_SIZE = 64           # Maximum allowed batch size

##############################
# GPU Utilization Helper Functions
##############################
def get_gpu_utilization(gpu_index=0):
    """Return current GPU utilization (%) by parsing nvidia-smi output."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        utilization_lines = result.strip().split("\n")
        utilization = float(utilization_lines[gpu_index])
        return utilization
    except Exception as e:
        print("Warning: Could not get GPU utilization:", e)
        return 0.0

def print_gpu_info(gpu_index=0):
    """Print GPU properties and memory stats using torch and nvidia-smi."""
    try:
        props = torch.cuda.get_device_properties(gpu_index)
        print(f"--- GPU Information ---")
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"Multiprocessor Count: {props.multi_processor_count}")
        util = get_gpu_utilization(gpu_index)
        print(f"Current GPU Utilization (nvidia-smi): {util:.1f}%")
        print(f"-----------------------")
    except Exception as e:
        print("Unable to get GPU properties:", e)

##############################
# Dataset, Tokenization, and Preprocessing
##############################
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
    """
    Process each sample to create a uniform format.
    Searches for common text keys and, if available, extra fields like
    'thinking', 'task', or 'function_call(s)' are appended to the main text.
    """
    out = {}
    # Find the primary text from one of several common keys.
    for key in ["question", "caption", "text", "code"]:
        if key in sample and sample[key] is not None:
            out["text"] = sample[key]
            break
    if "text" not in out:
        out["text"] = ""
    # Use answer if available; otherwise, repeat text as target.
    if "answer" in sample and sample["answer"]:
        out["target"] = sample["answer"]
    else:
        out["target"] = out["text"]
    # Pass along image data, if any.
    if "image" in sample and sample["image"] is not None:
        out["image"] = sample["image"]
    else:
        out["image"] = None

    # Append any extra reasoning fields if they exist.
    extra_fields = []
    for field in ["thinking", "task", "function_call", "function_calls"]:
        if field in sample and sample[field]:
            extra_fields.append(f"<{field.upper()}> {sample[field]}")
    if extra_fields:
        out["text"] += " " + " ".join(extra_fields)
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
    # Original dataset IDs plus additional reasoning/chain-of-thought datasets.
    dataset_ids = [
        "Multimodal-Fatima/VQAv2_sample_train",
        "Multimodal-Fatima/OxfordFlowers_test",
        "matlok/multimodal-python-copilot-training-overview",  # may not exist; will be skipped if error
        "notbadai/python_functions_reasoning",
        "espejelomar/code_search_net_python_10000_examples",
        "reshinthadith/synthetic_program_synthesis_python_1M",
        "suriyagunasekar/stackoverflow-python-with-meta-data",
        "Sridevi/python_textbooks",
        "nuprl/stack-dedup-python-testgen-starcoder-filter-v2",
        # New reasoning/chain-of-thought datasets:
        "nvidia/OpenCodeReasoning",
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "open-thoughts/OpenThoughts2-1M",
        "glaiveai/reasoning-v1-20m",
        "emilbiju/Execution-Dagger-Data-Math-think",
        "wikimedia/wikipedia",
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "facebook/natural_reasoning",
        "KingNish/reasoning-base-20k",
        "ProlificAI/social-reasoning-rlhf",
        "dvilasuero/natural-science-reasoning",
        "smirki/UI_Reasoning_Dataset",
        "reasoning-machines/gsm-hard",
        "di-zhang-fdu/R1-Vision-Reasoning-Instructions",
        "lightblue/reasoning-multilingual-R1-Llama-70B-train",
        "prithivMLmods/Deepthink-Reasoning",
        "Nan-Do/SPP_30K_reasoning_tasks",
        "davanstrien/reasoning-required",
        "antiven0m/physical-reasoning-dpo",
        "isaiahbjork/cot-logic-reasoning",
        "efficientscaling/Z1-Code-Reasoning-107K",
        "iamtarun/python_code_instructions_18k_alpaca",
        "flytech/python-codes-25k",
        "Vezora/Tested-143k-Python-Alpaca",
        "matlok/multimodal-python-copilot-training-overview",
        "semeru/code-text-python",
        "microsoft/LCC_python",
        "thomwolf/github-python",
        "Jofthomas/hermes-function-calling-thinking-V1",
        "UCSC-VLAA/VLAA-Thinking",
        "minchyeom/thinker-formatted",
        "fhai50032/GPQA-Thinking-O1",
        "ThinkAgents/Function-Calling-with-Chain-of-Thoughts",
        "Salesforce/xlam-function-calling-60k"
    ]
    processed_datasets = []
    for ds_id in dataset_ids:
        try:
            # For some datasets, if "OxfordFlowers" is mentioned, use split "test"; otherwise "train".
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

##############################
# Main Training Function
##############################
def main():
    try:
        # Enable cuDNN benchmarking for optimized GPU performance
        torch.backends.cudnn.benchmark = True

        # 1. Load and prepare the dataset
        dataset = load_and_prepare_datasets()

        # 2. Build the vocabulary and save it
        vocab = build_vocab(dataset, text_field="text", max_vocab_size=VOCAB_SIZE)
        with open("vocab.json", "w") as f:
            json.dump(vocab, f)
        print("Vocabulary built and saved to vocab.json.")

        # 3. Create a reverse vocabulary mapping
        rev_vocab = {str(idx): word for word, idx in vocab.items()}

        # 4. Setup initial DataLoader
        current_batch_size = INITIAL_BATCH_SIZE
        loader = DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, vocab),
            num_workers=8,
            pin_memory=True
        )

        # 5. Define the model configuration (must match training configuration)
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

        # 6. Setup device and load the model fully on GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print_gpu_info(torch.cuda.current_device())
        model = TitanModel(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Limit GPU memory (reserve ~10% headroom)
        if device.type == "cuda":
            device_index = torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(0.9, device=device_index)
            torch.cuda.empty_cache()
            print("GPU memory limited to 90% of total (10% reserved).")

        # 7. Setup Automatic Mixed Precision (AMP) training with safe fallback
        try:
            scaler = torch.amp.GradScaler(device_type=None, enabled=(device.type == 'cuda'))
            autocast_ctx = lambda: torch.amp.autocast(device_type=None, enabled=(device.type == 'cuda'))
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
            autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=(device.type == 'cuda'))

        # 8. Training loop with dynamic batch size adjustment
        model.train()
        num_epochs = 100
        epoch = 0
        while epoch < num_epochs:
            epoch += 1

            # Query GPU utilization and print info
            current_util = get_gpu_utilization()
            print(f"\nEpoch {epoch}: Current GPU Utilization: {current_util:.1f}%")
            
            # Adjust batch size if GPU utilization is below target
            if current_util < TARGET_GPU_UTILIZATION and device.type == "cuda":
                factor = TARGET_GPU_UTILIZATION / (current_util + 1e-6)
                new_bs = min(MAX_BATCH_SIZE, int(current_batch_size * factor))
                if new_bs > current_batch_size:
                    print(f"GPU utilization is low. Increasing batch size from {current_batch_size} to {new_bs}.")
                    current_batch_size = new_bs
                    loader = DataLoader(
                        dataset,
                        batch_size=current_batch_size,
                        shuffle=True,
                        collate_fn=lambda batch: collate_fn(batch, vocab),
                        num_workers=8,
                        pin_memory=True
                    )
            else:
                print("No batch size adjustment necessary for this epoch.")

            running_loss = 0.0
            batch_count = 0
            epoch_start_time = time.time()

            for batch in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} Progress", leave=False):
                try:
                    optimizer.zero_grad()
                    with autocast_ctx():
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

                except RuntimeError as e:
                    # Catch out-of-memory errors, reduce batch size, and continue.
                    if "out of memory" in str(e):
                        print("Out-of-memory error caught. Reducing batch size.")
                        current_batch_size = max(INITIAL_BATCH_SIZE, current_batch_size // 2)
                        loader = DataLoader(
                            dataset,
                            batch_size=current_batch_size,
                            shuffle=True,
                            collate_fn=lambda batch: collate_fn(batch, vocab),
                            num_workers=8,
                            pin_memory=True
                        )
                        torch.cuda.empty_cache()
                        break
                    else:
                        raise e

            epoch_duration = time.time() - epoch_start_time
            avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
            print(f"Epoch {epoch}/{num_epochs} completed in {epoch_duration:.1f}s - Average Loss: {avg_loss:.4f}")

            # 9. Quick validation check at epoch end
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

            if acc >= 100.0:
                print("Test passed 100%! Stopping training.")
                break

        print("\nTraining completed.")

    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
