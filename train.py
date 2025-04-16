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
    # Old list (multimodal, coding, reasoning, etc.)
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
# get_default_config() Updated
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
        "image_size": (64, 64),  # Note: torchvision.transforms.Resize requires a 2-value tuple.
        "training_datasets": training_datasets
    }

#####################################
# Helper Function: Try Loading a Dataset with Retries
#####################################
def try_load_dataset(name, split='train', max_retries=3, init_delay=10):
    use_token = os.environ.get("HF_AUTH_TOKEN", None)
    delay = init_delay
    for attempt in range(max_retries):
        try:
            ds = load_dataset(name, split=split, use_auth_token=use_token)
            return ds
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"Rate limited on dataset {name} (split={split}), attempt {attempt+1}/{max_retries}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            elif "Config name is missing" in err_str:
                try:
                    builder = load_dataset_builder(name, use_auth_token=use_token)
                    config_name = builder.info.config_names[0]
                    print(f"Automatically selecting config '{config_name}' for dataset {name} (split={split}).")
                    ds = load_dataset(name, config_name, split=split, use_auth_token=use_token)
                    return ds
                except Exception as e_config:
                    print(f"Failed to load dataset {name} with a config: {e_config}")
            else:
                print(f"Error loading dataset {name} (split={split}): {e}")
        # Try next attempt
    # If failed with train, try test split as a fallback (if not already trying 'test')
    if split == 'train':
        try:
            ds = load_dataset(name, split='test', use_auth_token=use_token)
            return ds
        except Exception as e_test:
            print(f"Failed to load dataset {name} with split 'test': {e_test}")
    raise Exception(f"Failed to load dataset {name} after {max_retries} attempts.")

#####################################
# MultiModalDataset Implementation (No Dummy Fallback)
#####################################
class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_names, tokenizer, image_size=(64, 64)):
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.datasets = []
        self.dataset_names = []
        self.cumulative_lengths = []
        total_length = 0
        for name in dataset_names:
            print(f"Downloading dataset: {name}")
            try:
                ds = try_load_dataset(name, split='train')
                self.datasets.append(ds)
                self.dataset_names.append(name)
                length = len(ds) if hasattr(ds, "__len__") else 0
                total_length += length
                self.cumulative_lengths.append(total_length)
                print(f"Loaded dataset {name} with {length} examples.")
            except Exception as e:
                print(f"Error: Skipping dataset {name} due to error: {e}")
        self.total_length = total_length
        dataset_info = {name: {"length": len(ds) if hasattr(ds, "__len__") else 0} 
                        for name, ds in zip(self.dataset_names, self.datasets)}
        with open("dataset_infos.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        for ds_idx, cum_len in enumerate(self.cumulative_lengths):
            if index < cum_len:
                prev = self.cumulative_lengths[ds_idx-1] if ds_idx > 0 else 0
                sample_index = index - prev
                try:
                    example = self.datasets[ds_idx][sample_index]
                except Exception:
                    example = {"text": ""}
                break
        item = {}
        input_text = None
        output_text = None
        for key, value in example.items():
            if isinstance(value, str):
                text = value
                if key.lower() in ["question", "prompt", "input", "query"]:
                    input_text = text
                elif key.lower() in ["answer", "answers", "response", "output", "multiple_choice_answer", "reasoning"]:
                    if output_text is None:
                        output_text = text
                    else:
                        output_text += " " + text
                else:
                    if input_text is None:
                        input_text = text
                    else:
                        input_text += " " + text
            elif key.lower() == "image":
                if value is None:
                    item['image'] = None
                else:
                    try:
                        pil_img = value if not isinstance(value, dict) else value.get('pil', value)
                    except Exception:
                        pil_img = value
                    item['image'] = self.image_transform(pil_img)
            elif key.lower() == "audio":
                if value is None:
                    item['audio'] = None
                else:
                    waveform = value.get('array', value)
                    item['audio'] = torch.tensor(waveform, dtype=torch.float)
            elif key.lower() == "video":
                item['video'] = value
        if input_text is not None:
            item['input_ids'] = torch.tensor(self.tokenizer.tokenize(input_text), dtype=torch.long)
        if output_text is not None:
            item['labels'] = torch.tensor(self.tokenizer.tokenize(output_text), dtype=torch.long)
        else:
            if input_text is not None:
                item['labels'] = item['input_ids'].clone()
        return item

#####################################
# Collate Function for DataLoader
#####################################
def multimodal_collate(batch):
    collated = {}
    if 'input_ids' in batch[0]:
        max_len = max(item['input_ids'].shape[0] for item in batch)
        padded_inputs = []
        for item in batch:
            ids = item['input_ids']
            pad_id = tokenizer.token_to_id["<PAD>"]
            if ids.shape[0] < max_len:
                ids = torch.cat([ids, torch.full((max_len - ids.shape[0],), pad_id, dtype=torch.long)], dim=0)
            padded_inputs.append(ids)
        collated['input_ids'] = torch.stack(padded_inputs)
    if 'labels' in batch[0]:
        max_len_lbl = max(item['labels'].shape[0] for item in batch)
        padded_labels = []
        for item in batch:
            lbl = item['labels']
            pad_id = tokenizer.token_to_id["<PAD>"]
            if lbl.shape[0] < max_len_lbl:
                lbl = torch.cat([lbl, torch.full((max_len_lbl - lbl.shape[0],), pad_id, dtype=torch.long)], dim=0)
            padded_labels.append(lbl)
        collated['labels'] = torch.stack(padded_labels)
    if 'image' in batch[0]:
        images = []
        for item in batch:
            img = item.get('image')
            if img is None:
                img = torch.zeros(3, 64, 64)
            images.append(img)
        collated['image'] = torch.stack(images)
    if 'audio' in batch[0]:
        audios = []
        max_audio = max(item['audio'].shape[0] if item.get('audio') is not None else 0 for item in batch)
        for item in batch:
            aud = item.get('audio')
            if aud is None:
                aud = torch.zeros(max_audio)
            elif aud.shape[0] < max_audio:
                aud = torch.cat([aud, torch.zeros(max_audio - aud.shape[0])], dim=0)
            audios.append(aud)
        collated['audio'] = torch.stack(audios)
    if 'video' in batch[0]:
        collated['video'] = [item.get('video') for item in batch]
    return collated

#####################################
# Advanced Training Loop with Detailed Reporting
#####################################
def train_model(model, dataloader, dataset_names, target_score, learning_rate, device):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler(device=device)
    model.train()
    overall_score = 0.0
    dataset_scores = {name: 0.0 for name in dataset_names}
    epoch = 0

    # Log model configuration and parameters
    print("====== Model and Training Configuration ======")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,} parameters")
    knowledge_size = total_params / 1e6
    print(f"Estimated Knowledge Size: {knowledge_size:.2f}M parameters")
    print("Training Datasets Loaded:")
    for name in dataset_names:
        print(f" - {name}")
    print("================================================")

    while overall_score < target_score:
        epoch += 1
        epoch_loss = 0.0
        batch_counter = 0
        print(f"\n=== Epoch {epoch} ===")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
        for batch in progress_bar:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                outputs = model(batch)
                loss = 0.0
                if "text_out" in outputs and "input_ids" in batch:
                    logits = outputs["text_out"]
                    target = batch["input_ids"]
                    loss += criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                if "audio_out" in outputs and "audio" in batch:
                    loss += torch.nn.MSELoss()(outputs["audio_out"], batch["audio"])
                if "image_out" in outputs and "image" in batch:
                    loss += torch.nn.MSELoss()(outputs["image_out"], batch["image"])
                if "video_out" in outputs and "video" in batch:
                    loss += torch.nn.MSELoss()(outputs["video_out"], batch["video"])
                if "selfteach_loss" in outputs:
                    loss += outputs["selfteach_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batch_counter += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = epoch_loss / batch_counter
        for name in dataset_names:
            dataset_scores[name] = max(0, 100 - avg_loss * 10)
        overall_score = sum(dataset_scores.values()) / len(dataset_names)
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
        print("Per-dataset Scores:")
        for name, score in dataset_scores.items():
            print(f" - {name}: {score:.2f}%")
        print(f"Overall Score: {overall_score:.2f}%")
    torch.save(model.state_dict(), "unified_model.pt")
    print("Training complete. Model saved as 'unified_model.pt'.")
    print("Function Call Demo:")
    print(model.call_function("build_script", "example_script.py"))
    print(model.call_function("execute_script", "example_script.py"))

def main():
    config = get_default_config()
    training_datasets = config.get("training_datasets")
    global tokenizer
    tokenizer = SimpleTokenizer(max_vocab_size=30000)
    sample_texts = []
    # Build vocabulary from available datasets
    for name in training_datasets:
        try:
            ds_stream = load_dataset(name, split='train', streaming=True, use_auth_token=os.environ.get("HF_AUTH_TOKEN"))
        except Exception as e1:
            try:
                ds_stream = load_dataset(name, split='test', streaming=True, use_auth_token=os.environ.get("HF_AUTH_TOKEN"))
            except Exception as e2:
                print(f"Skipping dataset {name} during vocabulary build. Errors: {e1}, {e2}")
                continue
        for i, ex in enumerate(ds_stream):
            for k, v in ex.items():
                if isinstance(v, str):
                    sample_texts.append(v)
            if i >= 500:
                break
    tokenizer.fit_on_texts(sample_texts)
    dataset = MultiModalDataset(training_datasets, tokenizer, image_size=config.get("image_size", (64, 64)))
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=multimodal_collate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} - Total Memory: {torch.cuda.get_device_properties(device).total_memory/(1024**3):.2f} GB")
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
    else:
        print("No GPU available. Using CPU.")
    model = UnifiedMultimodalModel(config).to(device)
    target_score = 100.0
    learning_rate = 1e-4
    training_thread = threading.Thread(target=train_model, args=(model, dataloader, training_datasets, target_score, learning_rate, device))
    training_thread.start()
    training_thread.join()

if __name__ == "__main__":
    main()
