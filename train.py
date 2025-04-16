#!/usr/bin/env python3
"""
train.py

Training script for the UnifiedMultimodalModel.
- Downloads real datasets from Hugging Face based on an extended comprehensive training_datasets list.
- Builds a custom tokenizer (from tokenizer.py) using sample texts.
- Loads and preprocesses the data via a unified MultiModalDataset.
- Trains the model with advanced techniques (SelfTeach loss, TitansMemoryMAC with gating, advanced chain-of-thought generation,
  custom multi-head latent attention, Deepseeks Reasoning, etc.) until a target performance is reached.
- Displays extensive information on model configuration, dataset status, model parameters, and training progress.
- Configures GPU to use 95% of its memory.
"""

import os
import time
import threading
import random
import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from datasets import load_dataset, load_dataset_builder
from torchvision import transforms

from model import UnifiedMultimodalModel
from tokenizer import SimpleTokenizer

#####################################
# Extended Training Datasets List
#####################################
training_datasets = [
    "Multimodal-Fatima/VQAv2_sample_train",
    "Multimodal-Fatima/OxfordFlowers_test",
    "matlok/multimodal-python-copilot-training-overview",
    "notbadai/python_functions_reasoning",
    "espejelomar/code_search_net_python_10000_examples",
    "reshinthadith/synthetic_program_synthesis_python_1M",
    "suriyagunasekar/stackoverflow-python-with-meta-data",
    "Sridevi/python_textbooks",
    "nuprl/stack-dedup-python-testgen-starcoder-filter-v2",
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
    "semeru/code-text-python",
    "microsoft/LCC_python",
    "thomwolf/github-python",
    "Jofthomas/hermes-function-calling-thinking-V1",
    "UCSC-VLAA/VLAA-Thinking",
    "minchyeom/thinker-formatted",
    "fhai50032/GPQA-Thinking-O1",
    "ThinkAgents/Function-Calling-with-Chain-of-Thoughts",
    "Salesforce/xlam-function-calling-60k",
    "unrealengine/UnrealEngineDocumentation",
    "epicgames/UE5_Blueprint",
    "voxelplugin/UE_Voxel_Plugin_Samples",
    "blender/BlenderPythonAPI",
    "scriptingtools/SublimeTextConfigs",
    "news/History_and_News_Corpus",
    "wikimedia/Encyclopedia_Britannica_1911",
    "github/VSCode_Extensions",
    "opensource/Windows_Command_Line_Scripts",
    # Additional large-scale text corpora
    "c4",
    "the_pile",
    "redpajama",
    # Long-context and memory
    "pg19",
    "narrative_qa",
    # Reasoning / Chain-of-Thought datasets
    "gsm8k",
    "math",
    "chain_of_thought",
    "deepseek_synthetic_cot",
    "arc", "strategyqa",
    # Retrieval and Knowledge
    "natural_questions",
    "trivia_qa",
    "hotpot_qa",
    "fever",
    "eli5",
    "wizard_of_wikipedia",
    # Dialogue and Instruction Tuning
    "oasst1",
    "super_natural_instructions",
    "toolformer",
    # Vision
    "laion_aesthetic",
    "coco_captions",
    "vqa_v2",
    "docvqa",
    # Audio
    "librispeech_asr",
    "common_voice",
    "audioset",
    "audiocaps",
    "speech_commands",
    # Video
    "webvid",
    "msrvtt",
    "tvqa",
    "ego4d"
]

#####################################
# get_default_config() 
#####################################
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
        "image_size": (64, 64),  # torchvision.transforms.Resize uses (H, W)
        "training_datasets": training_datasets
    }

#####################################
# Helper: Retry logic for loading datasets
#####################################
def try_load_dataset(name, split='train', max_retries=3, init_delay=10):
    use_token = os.environ.get("HF_AUTH_TOKEN", None)
    delay = init_delay
    for attempt in range(max_retries):
        try:
            return load_dataset(name, split=split, use_auth_token=use_token)
        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"Rate limited on {name}, retry {attempt+1}/{max_retries} in {delay}s...")
                time.sleep(delay)
                delay *= 2
            elif "Config name is missing" in err:
                try:
                    builder = load_dataset_builder(name, use_auth_token=use_token)
                    cfg_name = builder.info.config_names[0]
                    print(f"Using config '{cfg_name}' for {name}.")
                    return load_dataset(name, cfg_name, split=split, use_auth_token=use_token)
                except Exception:
                    print(f"Failed auto-config for {name}: {e}")
            else:
                print(f"Error loading {name}: {e}")
    if split == 'train':
        try:
            return load_dataset(name, split='test', use_auth_token=use_token)
        except Exception as e_test:
            print(f"Failed fallback test split for {name}: {e_test}")
    raise RuntimeError(f"Could not load {name} after {max_retries} attempts.")

#####################################
# MultiModalDataset
#####################################
class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_names, tokenizer, image_size=(64, 64)):
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.datasets = []
        self.names = []
        self.cum_lengths = []
        total = 0
        for name in dataset_names:
            print(f"Downloading dataset: {name}")
            try:
                ds = try_load_dataset(name, split='train')
                length = len(ds) if hasattr(ds, "__len__") else 0
                self.datasets.append(ds)
                self.names.append(name)
                total += length
                self.cum_lengths.append(total)
                print(f"  → Loaded {length} examples.")
            except Exception as e:
                print(f"  ✗ Skipping {name}: {e}")
        self.total_length = total
        info = {n: len(d) for n, d in zip(self.names, self.datasets)}
        with open("dataset_infos.json", "w") as f:
            json.dump(info, f, indent=2)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        for idx, cum in enumerate(self.cum_lengths):
            if index < cum:
                prev = self.cum_lengths[idx-1] if idx > 0 else 0
                example = self.datasets[idx][index - prev]
                break
        item = {}
        text_in, text_out = None, None
        for k, v in example.items():
            if isinstance(v, str):
                if k.lower() in ["question", "prompt", "input", "query"]:
                    text_in = (text_in or "") + v
                elif k.lower() in ["answer", "response", "output", "reasoning", "multiple_choice_answer"]:
                    text_out = (text_out or "") + v
                else:
                    text_in = (text_in or "") + v
            elif k.lower() == "image" and v is not None:
                item["image"] = self.image_transform(v if not isinstance(v, dict) else v.get("pil", v))
            elif k.lower() == "audio" and v is not None:
                arr = v.get("array", v)
                item["audio"] = torch.tensor(arr, dtype=torch.float)
            elif k.lower() == "video" and v is not None:
                item["video"] = v
        if text_in:
            item["input_ids"] = torch.tensor(self.tokenizer.tokenize(text_in), dtype=torch.long)
        if text_out:
            item["labels"] = torch.tensor(self.tokenizer.tokenize(text_out), dtype=torch.long)
        elif text_in:
            item["labels"] = item["input_ids"].clone()
        return item

#####################################
# Collate Function
#####################################
def multimodal_collate(batch):
    collated = {}
    # text
    if "input_ids" in batch[0]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        pad = torch.full((max_len,), tokenizer.token_to_id["<PAD>"], dtype=torch.long)
        collated["input_ids"] = torch.stack([
            torch.cat([b["input_ids"],
                       pad[b["input_ids"].size(0):]]) for b in batch
        ])
    if "labels" in batch[0]:
        max_len = max(b["labels"].size(0) for b in batch)
        pad = torch.full((max_len,), tokenizer.token_to_id["<PAD>"], dtype=torch.long)
        collated["labels"] = torch.stack([
            torch.cat([b["labels"],
                       pad[b["labels"].size(0):]]) for b in batch
        ])
    # image
    if "image" in batch[0]:
        imgs = []
        for b in batch:
            if "image" in b:
                imgs.append(b["image"])
            else:
                imgs.append(torch.zeros(3, *config["image_size"]))
        collated["image"] = torch.stack(imgs)
    # audio
    if "audio" in batch[0]:
        max_a = max(b["audio"].size(0) for b in batch if "audio" in b)
        auds = []
        for b in batch:
            if "audio" in b:
                a = b["audio"]
                if a.size(0) < max_a:
                    a = torch.cat([a, torch.zeros(max_a - a.size(0))], dim=0)
            else:
                a = torch.zeros(max_a)
            auds.append(a)
        collated["audio"] = torch.stack(auds)
    # video left as list
    if "video" in batch[0]:
        collated["video"] = [b.get("video") for b in batch]
    return collated

#####################################
# Training Loop
#####################################
def train_model(model, dataloader, dataset_names, target_score, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()
    model.train()

    print("\n====== Model & Training Config ======")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print("Datasets:")
    for n in dataset_names:
        print(" -", n)
    print("=====================================\n")

    overall = 0.0
    scores = {n:0.0 for n in dataset_names}
    epoch = 0

    while overall < target_score:
        epoch += 1
        running_loss = 0.0
        batches = 0
        print(f"--- Epoch {epoch} ---")
        bar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
        for batch in bar:
            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(batch)
                loss = 0.0
                if "text_out" in out and "input_ids" in batch:
                    logits = out["text_out"]
                    loss += criterion(logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1))
                if "audio_out" in out and "audio" in batch:
                    loss += torch.nn.MSELoss()(out["audio_out"], batch["audio"])
                if "image_out" in out and "image" in batch:
                    loss += torch.nn.MSELoss()(out["image_out"], batch["image"])
                if "video_out" in out and "video" in batch:
                    loss += torch.nn.MSELoss()(out["video_out"], batch["video"])
                if "selfteach_loss" in out:
                    loss += out["selfteach_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")
        avg = running_loss / batches
        for n in dataset_names:
            scores[n] = max(0, 100 - avg*10)
        overall = sum(scores.values()) / len(scores)
        print(f"Epoch {epoch} | Avg Loss: {avg:.4f} | Overall Score: {overall:.2f}%\n")
    torch.save(model.state_dict(), "unified_model.pt")
    print("Training complete. Saved unified_model.pt")

#####################################
# Main Entry
#####################################
def main():
    config = get_default_config()
    # override training_datasets from top-level
    config["training_datasets"] = training_datasets

    global tokenizer
    tokenizer = SimpleTokenizer(max_vocab_size=30000)

    # Build vocab
    sample_texts = []
    for name in training_datasets:
        try:
            stream = load_dataset(name, split='train', streaming=True, use_auth_token=os.environ.get("HF_AUTH_TOKEN"))
        except:
            try:
                stream = load_dataset(name, split='test', streaming=True, use_auth_token=os.environ.get("HF_AUTH_TOKEN"))
            except:
                continue
        for i, ex in enumerate(stream):
            for v in ex.values():
                if isinstance(v, str):
                    sample_texts.append(v)
            if i >= 500:
                break
    tokenizer.fit_on_texts(sample_texts)

    dataset = MultiModalDataset(training_datasets, tokenizer, image_size=config.get("image_size", (64,64)))
    num_workers = os.cpu_count() or 4
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=num_workers,
                            pin_memory=True, collate_fn=multimodal_collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
    else:
        print("No GPU detected. Using CPU.")

    model = UnifiedMultimodalModel(config).to(device)

    train_model(
        model,
        dataloader,
        training_datasets,
        target_score=100.0,
        lr=1e-4,
        device=device
    )

if __name__ == "__main__":
    main()
