#!/usr/bin/env python3
"""
train.py

Training script for the UnifiedMultimodalModel with robust Hugging Face dataset loading:
  - Centralized HF authentication (via `huggingface-cli login` or HF_HUB_TOKEN)
  - Interactive dataset & split selection (choose subset or all, train/test/both)
  - Automatic config discovery for multi-config datasets
  - Rate-limit backoff, caching (cache_dir + DownloadMode), and streaming fallback
  - MultiModalDataset and collate function for unified text/image/audio/video loading
  - Advanced training loop with mixed precision, gradient scaling, and SelfTeach loss
"""

import os
import time
import json
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torchvision import transforms

from datasets import load_dataset, get_dataset_config_names, DownloadMode
from huggingface_hub import HfFolder

from model import UnifiedMultimodalModel
from tokenizer import SimpleTokenizer

# === ENV SETUP ===
CACHE_DIR     = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))
HF_HUB_TOKEN  = HfFolder.get_token() or os.environ.get("HF_HUB_TOKEN")
assert HF_HUB_TOKEN, "Please `huggingface-cli login` or set HF_HUB_TOKEN."

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
        "training_datasets": []
    }

def choose_datasets():
    print("\nAvailable training datasets:")
    for i, name in enumerate(ALL_DATASETS, start=1):
        print(f"  {i:3d}. {name}")
    print("  all  → select every dataset above")
    choice = input("\nEnter indices (e.g. 1,4,7) or `all` [default=all]: ").strip()
    if not choice or choice.lower() == "all":
        return ALL_DATASETS
    sel = []
    for tok in choice.split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok)-1
            if 0 <= idx < len(ALL_DATASETS):
                sel.append(ALL_DATASETS[idx])
    if not sel:
        print("No valid selection; defaulting to all.\n")
        return ALL_DATASETS
    return sel

def try_load_dataset(name, split="train", max_retries=3, init_delay=10):
    delay = init_delay
    for attempt in range(1, max_retries+1):
        try:
            configs = get_dataset_config_names(name)
            if configs:
                return load_dataset(name, configs[0], split=split,
                                    cache_dir=CACHE_DIR,
                                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                    token=HF_HUB_TOKEN)
            else:
                return load_dataset(name, split=split,
                                    cache_dir=CACHE_DIR,
                                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                    token=HF_HUB_TOKEN)
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                print(f"Rate-limited on {name} (attempt {attempt}), sleeping {delay}s…")
                time.sleep(delay)
                delay *= 2
                continue
            if "Config name is missing" in msg or "missing" in msg:
                try:
                    return load_dataset(name, split=split,
                                        cache_dir=CACHE_DIR,
                                        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                        token=HF_HUB_TOKEN)
                except:
                    pass
            print(f"Error loading {name}: {e}")
            break
    print(f"Falling back to streaming for {name}::{split}")
    return load_dataset(name, split=split, streaming=True, token=HF_HUB_TOKEN)

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, entries, tokenizer, image_size=(64,64)):
        self.tokenizer = tokenizer
        self.image_tf  = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.datasets  = []
        self.names     = []
        self.cumlen    = []
        total = 0

        for entry in entries:
            name, split = entry.split("::")
            print(f"Loading {entry} …")
            try:
                ds = try_load_dataset(name, split=split)
                n  = len(ds) if hasattr(ds, "__len__") else 0
                self.datasets.append((entry, ds))
                self.names.append(entry)
                total += n
                self.cumlen.append(total)
                print(f"  → {n} examples")
            except Exception as e:
                print(f"  ✗ Skipping {entry}: {e}")

        self.total_length = total
        with open("dataset_infos.json", "w") as f:
            info = {entry: (len(ds) if hasattr(ds,"__len__") else None)
                    for entry, ds in self.datasets}
            json.dump(info, f, indent=2)

    def __len__(self): return self.total_length

    def __getitem__(self, idx):
        for i, cum in enumerate(self.cumlen):
            if idx < cum:
                prev_entry, ds = self.datasets[i]
                prev = self.cumlen[i-1] if i>0 else 0
                ex   = ds[idx-prev]
                break

        item = {}
        tin = tout = None
        for k,v in ex.items():
            kl = k.lower()
            if isinstance(v, str):
                if kl in ("question","prompt","input","query"):
                    tin = (tin or "") + v
                elif kl in ("answer","output","reasoning","response"):
                    tout = (tout or "") + v
                else:
                    tin = (tin or "") + v

            elif kl=="image" and v is not None:
                img = v if not isinstance(v,dict) else v.get("pil", v)
                item["image"] = self.image_tf(img)

            elif kl=="audio" and v is not None:
                arr = v.get("array",v)
                item["audio"] = torch.tensor(arr, dtype=torch.float)

            elif kl=="video" and v is not None:
                item["video"] = v

        if tin:
            item["input_ids"] = torch.tensor(self.tokenizer.tokenize(tin), dtype=torch.long)
        if tout:
            item["labels"]    = torch.tensor(self.tokenizer.tokenize(tout), dtype=torch.long)
        elif tin:
            item["labels"]    = item["input_ids"].clone()

        return item

def multimodal_collate(batch):
    coll = {}
    # text / labels
    if "input_ids" in batch[0]:
        maxlen = max(b["input_ids"].size(0) for b in batch)
        pad = batch[0]["input_ids"].new_full((maxlen,), 0)
        coll["input_ids"] = torch.stack([
            torch.cat([b["input_ids"], pad[b["input_ids"].size(0):]])
            for b in batch
        ])
    if "labels" in batch[0]:
        maxlen = max(b["labels"].size(0) for b in batch)
        pad = batch[0]["labels"].new_full((maxlen,), -100)
        coll["labels"] = torch.stack([
            torch.cat([b["labels"], pad[b["labels"].size(0):]])
            for b in batch
        ])
    # image
    if "image" in batch[0]:
        coll["image"] = torch.stack([b.get("image", torch.zeros_like(batch[0]["image"])) for b in batch])
    # audio
    if "audio" in batch[0]:
        max_a = max(b["audio"].size(0) for b in batch)
        auds = []
        for b in batch:
            a = b.get("audio", torch.zeros(max_a))
            if a.size(0) < max_a:
                a = torch.cat([a, torch.zeros(max_a - a.size(0))], dim=0)
            auds.append(a)
        coll["audio"] = torch.stack(auds)
    # video
    if "video" in batch[0]:
        coll["video"] = [b.get("video") for b in batch]
    return coll

def train_model(model, dataloader, entries, target_score, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler    = GradScaler()
    model.train()

    print("\n=== TRAINING CONFIG ===")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Datasets:")
    for e in entries:
        print("  -", e)
    print("=======================\n")

    overall, epoch = 0.0, 0
    amp_dev = "cuda" if torch.cuda.is_available() else "cpu"

    while overall < target_score:
        epoch += 1
        total_loss, batches = 0.0, 0
        print(f"--- Epoch {epoch} ---")
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            # move to device
            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            # build inputs for your model
            inputs = {}
            if "input_ids" in batch: inputs["text"]  = batch["input_ids"]
            if "audio"     in batch: inputs["audio"] = batch["audio"]
            if "image"     in batch: inputs["image"] = batch["image"]
            if "video"     in batch: inputs["video"] = batch["video"]
            # (no query during training)

            optimizer.zero_grad()
            with autocast(device_type=amp_dev):
                out  = model(inputs)
                loss = 0.0
                # text
                if "text_out" in out and "labels" in batch:
                    logits = out["text_out"]
                    loss  += criterion(logits.view(-1, logits.size(-1)),
                                       batch["labels"].view(-1))
                # audio
                if "audio_out" in out and "audio" in batch:
                    loss += torch.nn.MSELoss()(out["audio_out"], batch["audio"])
                # image
                if "image_out" in out and "image" in batch:
                    loss += torch.nn.MSELoss()(out["image_out"], batch["image"])
                # video
                if "video_out" in out and "video" in batch:
                    loss += torch.nn.MSELoss()(out["video_out"], batch["video"])
                # self-teach
                if "selfteach_loss" in out:
                    loss += out["selfteach_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batches    += 1

        avg     = total_loss / batches
        overall = max(0, 100 - avg * 10)
        print(f"Epoch {epoch} | Avg Loss: {avg:.4f} | Score: {overall:.2f}%\n")

    torch.save(model.state_dict(), "unified_model.pt")
    print("Training complete. Model saved to unified_model.pt")

def main():
    # 1) choose datasets
    training_datasets = choose_datasets()

    # 2) choose splits
    split_choice = input("Which split(s)? (train/test/both) [both]: ").strip().lower() or "both"
    entries = []
    for name in training_datasets:
        if split_choice in ("both","train"):
            entries.append(f"{name}::train")
        if split_choice in ("both","test"):
            entries.append(f"{name}::test")

    # 3) config
    cfg = get_default_config()
    cfg["training_datasets"] = training_datasets

    # 4) tokenizer warm-up
    tok    = SimpleTokenizer(max_vocab_size=30000)
    samples=[]
    for name in training_datasets:
        try:
            stream = try_load_dataset(name, split="train")
            for i,ex in enumerate(stream):
                for v in ex.values():
                    if isinstance(v,str):
                        samples.append(v)
                if i>=100: break
        except:
            continue
    tok.fit_on_texts(samples)

    # 5) prepare dataset + dataloader
    ds = MultiModalDataset(entries, tok, image_size=cfg["image_size"])
    dl = DataLoader(ds, batch_size=2, shuffle=True,
                    num_workers=os.cpu_count() or 4,
                    pin_memory=True, collate_fn=multimodal_collate)

    # 6) device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95, 0)

    # 7) train
    model = UnifiedMultimodalModel(cfg).to(device)
    train_model(model, dl, entries, target_score=100.0, lr=1e-4, device=device)

if __name__ == "__main__":
    main()
