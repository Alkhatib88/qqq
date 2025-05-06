# train.py

#!/usr/bin/env python3
"""
train.py

Training script for UnifiedMultimodalModel:
  - HF authentication & caching
  - CLI for picking datasets + splits
  - Robust HF dataset loading with fallback
  - Mixed‐precision training + SelfTeach loss
"""

import os, time, json, random
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

# ─── Environment setup ───────────────────────────────────────
CACHE_DIR     = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))
HF_HUB_TOKEN  = HfFolder.get_token() or os.environ.get("HF_HUB_TOKEN")
assert HF_HUB_TOKEN, "Please `huggingface-cli login` or set HF_HUB_TOKEN."

# ─── Available datasets list ─────────────────────────────────
ALL_DATASETS = [
    "Multimodal-Fatima/VQAv2_sample_train",
    "Multimodal-Fatima/OxfordFlowers_test",
    "notbadai/python_functions_reasoning",
    "espejelomar/code_search_net_python_10000_examples",
    "reshinthadith/synthetic_program_synthesis_python_1M",
    "suriyagunasekar/stackoverflow-python-with-meta-data",
    "Sridevi/python_textbooks",
    "nuprl/stack-dedup-python-testgen-starcoder-filter-v2",
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
    # extras (toggle on/off)
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
        "text_vocab_size":10000,"text_embed_dim":512,
        "text_encoder_layers":2,"text_decoder_layers":2,
        "text_num_heads":8,"text_ff_dim":1024,"text_max_len":128,
        "text_seq_len":32,
        "audio_latent_dim":256,"audio_output_length":16000,
        "image_latent_dim":256,
        "video_latent_dim":256,"video_num_frames":16,"video_frame_size":(64,64),
        "core_fused_dim":512,"external_fused_dim":512,
        "attention_num_heads":8,"attention_latent_dim":128,
        "cot_decoder_layers":2,"cot_max_len":128,
        "rag_documents":["Doc1","Doc2","Doc3"],
        "image_size":(64,64),
        "training_datasets":[]
    }

def choose_datasets():
    print("\nAvailable training datasets:")
    for i,name in enumerate(ALL_DATASETS,1):
        print(f"  {i:3d}. {name}")
    print("  all  → select every dataset above")
    choice = input("Select indices (e.g. 1,4,7) or ‘all’ [all]: ").strip().lower()
    if not choice or choice=="all":
        return ALL_DATASETS
    sel=[]
    for tok in choice.split(","):
        if tok.isdigit():
            idx=int(tok)-1
            if 0<=idx<len(ALL_DATASETS):
                sel.append(ALL_DATASETS[idx])
    if not sel:
        print("No valid; defaulting to all.")
        return ALL_DATASETS
    return sel

def try_load_dataset(name, split="train", retries=3, delay=5):
    for attempt in range(retries):
        try:
            cfgs = get_dataset_config_names(name)
            if cfgs:
                return load_dataset(name, cfgs[0], split=split,
                                    cache_dir=CACHE_DIR,
                                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                    token=HF_HUB_TOKEN)
            return load_dataset(name, split=split,
                                cache_dir=CACHE_DIR,
                                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                token=HF_HUB_TOKEN)
        except Exception as e:
            msg=str(e)
            if "Unknown split" in msg or "missing" in msg:
                # try streaming fallback
                break
            if "429" in msg:
                time.sleep(delay)
                delay*=2
                continue
            print(f"Error loading {name}: {e}")
            break
    print(f"Falling back to streaming for {name}::{split}")
    return load_dataset(name, split=split, streaming=True, token=HF_HUB_TOKEN)

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, entries, tokenizer, image_size=(64,64)):
        """
        entries: list of (dataset_name, split) tuples
        """
        self.tokenizer = tokenizer
        self.image_tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.datasets, self.names, self.splits, self.cumlen = [], [], [], []
        total=0
        for name,split in entries:
            print(f"  Loading {name}::{split} …")
            ds = try_load_dataset(name, split=split)
            n  = len(ds) if hasattr(ds,"__len__") else 0
            self.datasets.append(ds); self.names.append(name); self.splits.append(split)
            total+=n; self.cumlen.append(total)
            print(f"    → {n} examples")
        self.total_length = total

    def __len__(self): return self.total_length

    def __getitem__(self, idx):
        # find which sub-dataset
        for i,c in enumerate(self.cumlen):
            if idx < c:
                prev = self.cumlen[i-1] if i>0 else 0
                ex = self.datasets[i][idx-prev]
                break
        item,tin,tout={},None,None
        for k,v in ex.items():
            kl=k.lower()
            if isinstance(v,str):
                if kl in ("question","prompt","input","query"):
                    tin = (tin or "") + v
                elif kl in ("answer","output","reasoning","response"):
                    tout= (tout or "") + v
                else:
                    tin = (tin or "") + v
            elif kl=="image" and v is not None:
                img = v if not isinstance(v,dict) else v.get("pil",v)
                item["image"]=self.image_tf(img)
            elif kl=="audio" and v is not None:
                arr = v.get("array",v)
                item["audio"]=torch.tensor(arr,dtype=torch.float)
            elif kl=="video" and v is not None:
                item["video"]=v
        if tin:
            item["input_ids"]=torch.tensor(self.tokenizer.tokenize(tin),dtype=torch.long)
        if tout:
            item["labels"]=torch.tensor(self.tokenizer.tokenize(tout),dtype=torch.long)
        elif tin:
            item["labels"]=item["input_ids"].clone()
        return item

def multimodal_collate(batch):
    collated={}
    # text
    if "input_ids" in batch[0]:
        maxl=max(b["input_ids"].size(0) for b in batch)
        pad=batch[0]["input_ids"].new_full((maxl,),0)
        collated["input_ids"]=torch.stack([
            torch.cat([b["input_ids"],pad[b["input_ids"].size(0):]])
            for b in batch
        ])
    if "labels" in batch[0]:
        maxl=max(b["labels"].size(0) for b in batch)
        pad=batch[0]["labels"].new_full((maxl,),-100)
        collated["labels"]=torch.stack([
            torch.cat([b["labels"],pad[b["labels"].size(0):]])
            for b in batch
        ])
    # image
    if "image" in batch[0]:
        collated["image"]=torch.stack([b.get("image",torch.zeros_like(batch[0]["image"])) for b in batch])
    # audio
    if "audio" in batch[0]:
        maxa=max(b["audio"].size(0) for b in batch)
        auds=[]
        for b in batch:
            a=b.get("audio",torch.zeros(maxa))
            if a.size(0)<maxa:
                a=torch.cat([a,torch.zeros(maxa-a.size(0))],0)
            auds.append(a)
        collated["audio"]=torch.stack(auds)
    # video (list)
    if "video" in batch[0]:
        collated["video"]=[b.get("video") for b in batch]
    return collated

def train_model(model, dataloader, names, target_score, lr, device):
    opt   = Adam(model.parameters(), lr=lr)
    crit  = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler= GradScaler()
    model.train()
    print("\n=== TRAINING CONFIG ===")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    for n in names: print(" -",n)
    print("=======================\n")
    overall=0; epoch=0
    while overall < target_score:
        epoch+=1; total_loss=0.0; batches=0
        print(f"--- Epoch {epoch} ---")
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            for k,v in batch.items():
                if torch.is_tensor(v): batch[k]=v.to(device,non_blocking=True)
            opt.zero_grad()
            with autocast():
                out=model(batch)
                loss=0.0
                if "text_out" in out and "input_ids" in batch:
                    logits=out["text_out"]
                    loss+=crit(logits.view(-1,logits.size(-1)), batch["input_ids"].view(-1))
                if "audio_out" in out and "audio" in batch:
                    loss+=torch.nn.MSELoss()(out["audio_out"], batch["audio"])
                if "image_out" in out and "image" in batch:
                    loss+=torch.nn.MSELoss()(out["image_out"], batch["image"])
                if "video_out" in out and "video" in batch:
                    loss+=torch.nn.MSELoss()(out["video_out"], batch["video"])
                if "selfteach_loss" in out:
                    loss+=out["selfteach_loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss+=loss.item()
            batches+=1
        avg=total_loss/batches
        overall=max(0,100-avg*10)
        print(f"Epoch {epoch} | Avg Loss: {avg:.4f} | Score: {overall:.2f}%\n")
    torch.save(model.state_dict(),"unified_model.pt")
    print("Training complete. Model saved to unified_model.pt")

def main():
    # 1) choose
    names = choose_datasets()
    split = input("Which split(s)? (train/test/both) [both]: ").strip().lower() or "both"

    entries=[]
    if split in ("train","both"):
        entries += [(n, "train") for n in names]
    if split in ("test","both"):
        entries += [(n, "test")  for n in names]

    # 2) build config
    cfg = get_default_config()
    cfg["training_datasets"] = names

    # 3) build tokenizer from small sample of train
    tk = SimpleTokenizer(max_vocab_size=30000)
    samples=[]
    for n in names:
        try:
            ds = try_load_dataset(n, split="train")
            for i,ex in enumerate(ds):
                for v in ex.values():
                    if isinstance(v,str): samples.append(v)
                if i>=100: break
        except: pass
    tk.fit_on_texts(samples)

    # 4) dataset & dataloader
    dataset   = MultiModalDataset(entries, tk, image_size=cfg["image_size"])
    dataloader= DataLoader(dataset, batch_size=2, shuffle=True,
                           num_workers=os.cpu_count() or 4,
                           pin_memory=True, collate_fn=multimodal_collate)

    # 5) device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95,0)

    # 6) model & train
    model= UnifiedMultimodalModel(cfg).to(dev)
    train_model(model, dataloader, names, target_score=100.0, lr=1e-4, device=dev)

if __name__=="__main__":
    main()
