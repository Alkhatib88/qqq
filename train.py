#!/usr/bin/env python3
"""
train.py

Training script for the UnifiedMultimodalModel.
- Downloads real datasets from Hugging Face based on an extended comprehensive training_datasets list.
- Builds a custom tokenizer (from tokenizer.py) using sample texts.
- Loads and preprocesses the data via a unified MultiModalDataset.
- Trains the model with advanced techniques until a target performance is reached.
- Skips any dataset that errors and logs the exact exception.
"""

import os
import time
import threading
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
    # multimodal, coding, reasoning, etc.
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
    "c4", "the_pile", "redpajama",
    # Long-context and memory
    "pg19", "narrative_qa",
    # Reasoning / Chain-of-Thought datasets
    "gsm8k","math","chain_of_thought","deepseek_synthetic_cot",
    "arc","strategyqa",
    # Retrieval and Knowledge
    "natural_questions","trivia_qa","hotpot_qa","fever","eli5","wizard_of_wikipedia",
    # Dialogue and Instruction Tuning
    "oasst1","super_natural_instructions","toolformer",
    # Vision
    "laion_aesthetic","coco_captions","vqa_v2","docvqa",
    # Audio
    "librispeech_asr","common_voice","audioset","audiocaps","speech_commands",
    # Video
    "webvid","msrvtt","tvqa","ego4d"
]

#####################################
# get_default_config()
#####################################
def get_default_config():
    return {
        # ... all your hyperparams ...
        "training_datasets": training_datasets,
        "image_size": (64, 64),
    }

#####################################
# Helper: try_load_dataset with retries, auto‑config, throttling
#####################################
def try_load_dataset(name, split='train', max_retries=3, init_delay=5):
    delay = init_delay
    token = os.environ.get("HF_AUTH_TOKEN", None)
    for attempt in range(max_retries):
        try:
            return load_dataset(name, split=split, use_auth_token=token)
        except Exception as e:
            err = str(e)
            # rate‑limit
            if "429" in err:
                print(f"[429] {name}#{split} attempt {attempt+1}/{max_retries}, retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
                continue
            # pick first config if missing
            if "Config name is missing" in err or "Please pick one among the available configs" in err:
                try:
                    builder = load_dataset_builder(name, use_auth_token=token)
                    cfg = builder.info.config_name or builder.info.config_names[0]
                    print(f"[AUTO‑CONFIG] using '{cfg}' for {name}#{split}")
                    return load_dataset(name, cfg, split=split, use_auth_token=token)
                except Exception as e2:
                    print(f"[FAIL‑CFG] {name}#{split} → {e2}")
                    break
            # fallback to test split
            if split=='train':
                print(f"[FALLBACK] trying {name}#test due to {err}")
                return try_load_dataset(name,'test',max_retries,init_delay)
            # unrecoverable
            raise

#####################################
# MultiModalDataset (skips broken + logs)
#####################################
class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_names, tokenizer, image_size=(64,64)):
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.datasets, self.names = [], []
        self.cumlen = []
        total = 0

        for name in dataset_names:
            print(f"[DOWNLOADING] {name}")
            try:
                ds = try_load_dataset(name, split='train')
                length = len(ds)
                total += length
                self.datasets.append(ds)
                self.names.append(name)
                self.cumlen.append(total)
                print(f"[OK] {name} → {length} examples")
            except Exception as e:
                print(f"[SKIP] {name} → {e}")

        self.total = total

        # emit a JSON report
        info = {n: len(d) for n,d in zip(self.names,self.datasets)}
        with open("dataset_infos.json","w") as f:
            json.dump(info,f,indent=2)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # locate which dataset
        for i,cum in enumerate(self.cumlen):
            if idx<cum:
                prev = self.cumlen[i-1] if i>0 else 0
                ex = self.datasets[i][idx-prev]
                break
        else:
            ex = {}
        # simple multimodal parsing...
        item = {}
        text_fields = [v for k,v in ex.items() if isinstance(v,str)]
        text = " ".join(text_fields)[:512]
        item['input_ids'] = torch.tensor(self.tokenizer.tokenize(text))
        item['labels']    = item['input_ids'].clone()
        # images/audio/video can be added similarly
        return item

#####################################
# Collate
#####################################
def multimodal_collate(batch):
    # pad input_ids & labels
    max_len = max(x['input_ids'].size(0) for x in batch)
    pad_id  = tokenizer.token_to_id["<PAD>"]
    inputs  = []
    labels  = []
    for x in batch:
        ids = x['input_ids']
        pad = torch.full((max_len-ids.size(0),),pad_id,dtype=torch.long)
        inputs.append(torch.cat([ids,pad]))
        lbl = x['labels']
        pad2 = torch.full((max_len-lbl.size(0),),pad_id,dtype=torch.long)
        labels.append(torch.cat([lbl,pad2]))
    return {
        "input_ids": torch.stack(inputs),
        "labels":    torch.stack(labels)
    }

#####################################
# Training routine
#####################################
def train_model(model, dataloader, names, target_score, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler    = GradScaler()
    model.train()

    scores = {n:0.0 for n in names}
    overall = 0.0
    epoch = 0

    print("=== MODEL & DATASETS ===")
    print(f" Params: {sum(p.numel() for p in model.parameters()):,}")
    for n in names: print(f"  - {n}")
    print("=========================")

    while overall < target_score:
        epoch += 1
        total_loss = 0.0
        cnt = 0
        bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in bar:
            for k in batch: batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(batch)
                logits = out["text_out"]  # adjust to your model output
                loss   = criterion(logits.view(-1,logits.size(-1)), batch["labels"].view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            cnt += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss/cnt
        overall = max(0,100 - avg*10)
        print(f"Epoch {epoch} done — avg_loss {avg:.4f} — overall_score {overall:.2f}%")

    torch.save(model.state_dict(),"unified_model.pt")
    print("🚀 Training complete.")

def main():
    cfg = get_default_config()
    toks = SimpleTokenizer(max_vocab_size=30000)

    # Build vocab from a small sample of each dataset
    samples = []
    for name in cfg["training_datasets"]:
        try:
            ds = try_load_dataset(name, split='train', max_retries=1)
            for i,ex in enumerate(ds):
                for v in ex.values():
                    if isinstance(v,str):
                        samples.append(v)
                if i>200: break
        except Exception as e:
            print(f"[VOCAB SKIP] {name} → {e}")
    toks.fit_on_texts(samples)

    # Build our multimodal dataset
    ds = MultiModalDataset(cfg["training_datasets"], toks, image_size=cfg["image_size"])
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=multimodal_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.cuda.set_per_process_memory_fraction(0.95,0)

    model = UnifiedMultimodalModel(cfg).to(device)
    train_model(model, loader, ds.names, target_score=95.0, lr=1e-4, device=device)

if __name__=="__main__":
    main()
