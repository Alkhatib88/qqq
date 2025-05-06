#!/usr/bin/env python3
"""
train.py

Training script for the UnifiedMultimodalModel with robust Hugging Face dataset loading:
  - Centralized HF authentication (via `huggingface-cli login` or HF_HUB_TOKEN)
  - Interactive dataset & split selection (choose subset or all, train/test/both)
  - Skip & log any dataset/split that errors out
  - Automatic config discovery for multi-config datasets
  - Rate-limit backoff, caching (cache_dir + DownloadMode), and streaming fallback
  - MultiModalDataset and collate function for unified text/image/audio/video loading
  - Advanced training loop with mixed precision, gradient scaling, and SelfTeach loss
"""

import os
import time
import json
import random
import logging

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torchvision import transforms

from datasets import load_dataset, get_dataset_config_names, DownloadMode
from huggingface_hub import HfFolder

from model import UnifiedMultimodalModel
from tokenizer import SimpleTokenizer

# === Setup logging for dataset load errors ===
logging.basicConfig(
    filename="dataset_load.log",
    filemode="w",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s:%(message)s"
)

# === Constants & Environment Setup ===
CACHE_DIR = os.environ.get(
    "HF_HOME",
    os.path.expanduser("~/.cache/huggingface/datasets")
)
HF_HUB_TOKEN = HfFolder.get_token() or os.environ.get("HF_HUB_TOKEN")
assert HF_HUB_TOKEN, \
       "Please login via `huggingface-cli login` or set the HF_HUB_TOKEN env var."

# === All Available Datasets ===
ALL_DATASETS = [
    # core Python reasoning/code corpora
    "Multimodal-Fatima/VQAv2_sample_train",
    "Multimodal-Fatima/OxfordFlowers_test",
    "notbadai/python_functions_reasoning",
    "espejelomar/code_search_net_python_10000_examples",
    "reshinthadith/synthetic_program_synthesis_python_1M",
    "suriyagunasekar/stackoverflow-python-with-meta-data",
    "Sridevi/python_textbooks",
    "nuprl/stack-dedup-python-testgen-starcoder-filter-v2",
    # multimodal & reasoning
    "nvidia/OpenCodeReasoning",
    "open-thoughts/OpenThoughts2-1M",
    "glaiveai/reasoning-v1-20m",
    "emilbiju/Execution-Dagger-Data-Math-think",
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
    "semeru/code-text-python",
    "microsoft/LCC_python",
    "thomwolf/github-python",
    "Jofthomas/hermes-function-calling-thinking-V1",
    "minchyeom/thinker-formatted",
    "fhai50032/GPQA-Thinking-O1",
    "ThinkAgents/Function-Calling-with-Chain-of-Thoughts",
    # gated or extra (toggle as desired)
    "matlok/multimodal-python-copilot-training-overview",
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "wikimedia/wikipedia",
    "unrealengine/UnrealEngineDocumentation",
    "epicgames/UE5_Blueprint",
    "voxelplugin/UE_Voxel_Plugin_Samples",
    "blender/BlenderPythonAPI",
    "scriptingtools/SublimeTextConfigs",
    "wikimedia/Encyclopedia_Britannica_1911",
    "opensource/Windows_Command_Line_Scripts",
]

def get_default_config():
    """Returns the model configuration dict."""
    return {
        "text_vocab_size": 10000,
        "text_embed_dim": 512,
        "text_encoder_layers": 2,
        "text_decoder_layers": 2,
        "text_num_heads": 8,
        "text_ff_dim": 1024,
        "text_max_len": 128,
        "text_seq_len": 32,
        "audio_latent_dim": 256,
        "audio_output_length": 16000,
        "image_latent_dim": 256,
        "video_latent_dim": 256,
        "video_num_frames": 16,
        "video_frame_size": (64, 64),
        "core_fused_dim": 512,
        "external_fused_dim": 512,
        "attention_num_heads": 8,
        "attention_latent_dim": 128,
        "cot_decoder_layers": 2,
        "cot_max_len": 128,
        "rag_documents": [
            "Document 1: Advanced multimodal techniques.",
            "Document 2: Chain-of-thought reasoning improves performance.",
            "Document 3: Retrieval augmented generation in AI."
        ],
        "image_size": (64, 64),
        # will be overwritten in main()
        "training_datasets": []
    }

def choose_datasets():
    """Prompt the user to pick a subset of ALL_DATASETS (or all)."""
    print("\nAvailable training datasets:")
    for i, name in enumerate(ALL_DATASETS, start=1):
        print(f"  {i:3d}. {name}")
    print("  all  → select every dataset above")

    choice = input("\nSelect indices (e.g. 1,4,7) or ‘all’ [all]: ").strip()
    if not choice or choice.lower() == "all":
        return ALL_DATASETS

    selected = []
    for token in choice.split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(ALL_DATASETS):
                selected.append(ALL_DATASETS[idx])
    if not selected:
        print("No valid selection detected; defaulting to all datasets.\n")
        return ALL_DATASETS
    return selected

def try_load_dataset(name, split="train", max_retries=3, init_delay=10):
    """
    Load a dataset with:
      - config discovery via get_dataset_config_names()
      - local caching (CACHE_DIR)
      - rate-limit backoff
      - streaming fallback
    """
    delay = init_delay
    for attempt in range(1, max_retries + 1):
        try:
            configs = get_dataset_config_names(name)
            if configs:
                # use first config if multiple
                cfg = configs[0]
                return load_dataset(
                    name, cfg, split=split,
                    cache_dir=CACHE_DIR,
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                    token=HF_HUB_TOKEN
                )
            else:
                return load_dataset(
                    name, split=split,
                    cache_dir=CACHE_DIR,
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                    token=HF_HUB_TOKEN
                )
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                print(f"Rate limited on {name}::{split} (attempt {attempt}/{max_retries}), sleeping {delay}s…")
                time.sleep(delay)
                delay *= 2
                continue
            # Try fallback without explicit config
            if "missing" in msg.lower():
                try:
                    return load_dataset(
                        name, split=split,
                        cache_dir=CACHE_DIR,
                        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                        token=HF_HUB_TOKEN
                    )
                except:
                    pass
            # otherwise bail out
            raise

    # final fallback: streaming mode
    return load_dataset(
        name, split=split,
        streaming=True,
        token=HF_HUB_TOKEN
    )

class MultiModalDataset(torch.utils.data.Dataset):
    """
    Wraps multiple HF datasets (with explicit splits) into one unified torch Dataset.
    Skips & logs any (name,split) that errors out.
    """
    def __init__(self, dataset_splits, tokenizer, image_size=(64,64)):
        """
        dataset_splits: list of (name:str, split:str) tuples
        """
        self.tokenizer = tokenizer
        self.image_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self.datasets = []
        self.names    = []    # store "name::split"
        self.cumlen   = []
        total = 0

        for name, split in dataset_splits:
            tag = f"{name}::{split}"
            print(f"Loading {tag}…")
            try:
                ds = try_load_dataset(name, split=split)
                # if HF dataset supports __len__, use that; else attempt to pull first 1k
                n = len(ds) if hasattr(ds, "__len__") else None
                if n is None:
                    # streaming without length -> attempt to buffer first 1000
                    n = sum(1 for _ in ds.take(1000))
                self.datasets.append(ds)
                self.names.append(tag)
                total += n
                self.cumlen.append(total)
                print(f"  → {n} examples")
            except Exception as e:
                msg = str(e).replace("\n"," ")
                print(f"  ✗ Skipping {tag}: {msg}")
                logging.error(f"Skipping {tag}: {msg}")
                continue

        self.total_length = total
        # record final dataset sizes
        with open("dataset_infos.json", "w") as f:
            info = {n: (self.cumlen[i] - (self.cumlen[i-1] if i>0 else 0))
                    for i,n in enumerate(self.names)}
            json.dump(info, f, indent=2)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # find which sub‐dataset
        for i, cum in enumerate(self.cumlen):
            if idx < cum:
                prev = self.cumlen[i-1] if i>0 else 0
                ex = self.datasets[i][idx - prev]
                break
        else:
            raise IndexError(idx)

        item = {}
        tin = tout = None

        # unify text / image / audio / video fields
        for k,v in ex.items():
            kl = k.lower()
            if isinstance(v, str):
                if kl in ("question","prompt","input","query"):
                    tin = (tin or "") + v
                elif kl in ("answer","output","reasoning","response"):
                    tout = (tout or "") + v
                else:
                    tin = (tin or "") + v

            elif kl == "image" and v is not None:
                img = v if not isinstance(v, dict) else v.get("pil", v)
                item["image"] = self.image_tf(img)

            elif kl == "audio" and v is not None:
                arr = v.get("array", v)
                item["audio"] = torch.tensor(arr, dtype=torch.float)

            elif kl == "video" and v is not None:
                item["video"] = v

        if tin:
            item["input_ids"] = torch.tensor(
                self.tokenizer.tokenize(tin), dtype=torch.long
            )
        if tout:
            item["labels"] = torch.tensor(
                self.tokenizer.tokenize(tout), dtype=torch.long
            )
        elif tin:
            # auto-target = input (for auto-encoders)
            item["labels"] = item["input_ids"].clone()

        return item

def multimodal_collate(batch):
    """Pad & stack text, image, audio, leave video as list."""
    collated = {}

    # text / labels
    if "input_ids" in batch[0]:
        maxlen = max(b["input_ids"].size(0) for b in batch)
        pad = batch[0]["input_ids"].new_full((maxlen,), fill_value=0)
        collated["input_ids"] = torch.stack([
            torch.cat([b["input_ids"], pad[b["input_ids"].size(0):]])
            for b in batch
        ])
    if "labels" in batch[0]:
        maxlen = max(b["labels"].size(0) for b in batch)
        pad = batch[0]["labels"].new_full((maxlen,), fill_value=-100)
        collated["labels"] = torch.stack([
            torch.cat([b["labels"], pad[b["labels"].size(0):]])
            for b in batch
        ])

    # images
    if "image" in batch[0]:
        collated["image"] = torch.stack([
            b.get("image", torch.zeros_like(batch[0]["image"]))
            for b in batch
        ])

    # audio
    if "audio" in batch[0]:
        max_a = max(b["audio"].size(0) for b in batch)
        auds = []
        for b in batch:
            a = b.get("audio", torch.zeros(max_a))
            if a.size(0) < max_a:
                a = torch.cat([a, torch.zeros(max_a - a.size(0))], dim=0)
            auds.append(a)
        collated["audio"] = torch.stack(auds)

    # video
    if "video" in batch[0]:
        collated["video"] = [b.get("video") for b in batch]

    return collated

def train_model(model, dataloader, dataset_tags, target_score, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()
    model.train()

    print("\n=== TRAINING CONFIG ===")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Datasets:")
    for t in dataset_tags:
        print("  -", t)
    print("=======================\n")

    overall = 0.0
    epoch = 0

    while overall < target_score:
        epoch += 1
        total_loss = 0.0
        batches = 0
        print(f"--- Epoch {epoch} ---")
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            # move to device
            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                out = model(batch)
                loss = 0.0

                # text reconstruction
                if "text_out" in out and "input_ids" in batch:
                    logits = out["text_out"]
                    loss += criterion(
                        logits.view(-1, logits.size(-1)),
                        batch["input_ids"].view(-1)
                    )

                # audio / image / video MSE losses
                if "audio_out" in out and "audio" in batch:
                    loss += torch.nn.MSELoss()(out["audio_out"], batch["audio"])
                if "image_out" in out and "image" in batch:
                    loss += torch.nn.MSELoss()(out["image_out"], batch["image"])
                if "video_out" in out and "video" in batch:
                    loss += torch.nn.MSELoss()(out["video_out"], batch["video"])

                # self-teach
                if "selfteach_loss" in out:
                    loss += out["selfteach_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batches += 1

        avg = total_loss / batches
        overall = max(0, 100 - avg * 10)
        print(f"Epoch {epoch} | Avg Loss: {avg:.4f} | Score: {overall:.2f}%\n")

    torch.save(model.state_dict(), "unified_model.pt")
    print("Training complete. Model saved to unified_model.pt")

def main():
    # 1) Choose datasets
    training_datasets = choose_datasets()

    # 2) Ask splits
    split_choice = input("Which split(s)? (train/test/both) [both]: ").strip().lower() or "both"
    if split_choice not in ("train","test","both"):
        split_choice = "both"
    split_list = ["train"] if split_choice=="train" else \
                 ["test"]  if split_choice=="test"  else \
                 ["train","test"]

    # 3) Build the explicit (name,split) list
    dataset_splits = []
    for name in training_datasets:
        for sp in split_list:
            dataset_splits.append((name, sp))

    # 4) Build config
    cfg = get_default_config()
    cfg["training_datasets"] = training_datasets

    # 5) Build tokenizer via a small streaming sample
    tokenizer = SimpleTokenizer(max_vocab_size=30000)
    samples = []
    for name in training_datasets:
        try:
            # only sample from train
            ds = try_load_dataset(name, split="train")
            for i, ex in enumerate(ds):
                for v in ex.values():
                    if isinstance(v, str):
                        samples.append(v)
                if i >= 100:
                    break
        except:
            continue
    tokenizer.fit_on_texts(samples)

    # 6) Prepare dataset + dataloader
    dataset = MultiModalDataset(dataset_splits, tokenizer,
                                image_size=cfg.get("image_size", (64,64)))
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        num_workers=os.cpu_count() or 4,
        pin_memory=True, collate_fn=multimodal_collate
    )

    # 7) Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95, 0)

    # 8) Instantiate & train
    model = UnifiedMultimodalModel(cfg).to(device)
    train_model(model, dataloader, dataset.names,
                target_score=100.0, lr=1e-4, device=device)

if __name__ == "__main__":
    main()
