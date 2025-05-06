# model.py

#!/usr/bin/env python3
"""
model.py

UnifiedMultimodalModel: A fully integrated multimodal encoder-decoder with all advanced modules from scratch:
  - Rotary Positional Encoding
  - Custom KV Cache
  - TitansMemoryMAC with gating (Memory as a Context)
  - SelfTeachModule for reinforcement learning-based self-teaching
  - Advanced Chain-of-Thought Generator (multiple sampling with log-probabilities)
  - Custom Multi-Head Latent Attention (MLA) module
  - Deepseeks Reasoning module
  - Standard image, audio, video encoders/decoders and fusion modules
  - Function Call Interface
"""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################
# Rotary Positional Encoding
#####################################
def apply_rotary_positional_encoding(x, seq_dim=1):
    dim = x.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    positions = torch.arange(x.size(seq_dim), device=x.device).float()
    sinusoid = torch.einsum("i , j -> ij", positions, inv_freq)
    sin = sinusoid.sin()[None, :, :]
    cos = sinusoid.cos()[None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

#####################################
# RMSNorm
#####################################
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps
        return self.weight * (x / norm)

#####################################
# KV Cache Implementation
#####################################
class KVCache:
    def __init__(self):
        self.cache = {}
    def add(self, layer, key, value):
        if layer not in self.cache:
            self.cache[layer] = {"key": key, "value": value}
        else:
            self.cache[layer]["key"] = torch.cat([self.cache[layer]["key"], key], dim=1)
            self.cache[layer]["value"] = torch.cat([self.cache[layer]["value"], value], dim=1)
    def get(self, layer):
        return self.cache.get(layer, {"key": None, "value": None})
    def reset(self):
        self.cache = {}

#####################################
# TitansMemoryMAC with Gating (Memory as a Context)
#####################################
class TitansMemoryMAC(nn.Module):
    def __init__(self, memory_size, embedding_dim, num_heads=4):
        super(TitansMemoryMAC, self).__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.gate_fc = nn.Linear(embedding_dim * 2, embedding_dim)
    def forward(self, x):
        query = x.unsqueeze(1)  # (batch, 1, embedding_dim)
        key = self.memory.unsqueeze(0).expand(x.size(0), -1, -1)
        value = key
        attn_output, _ = self.attn(query, key, value)
        attn_output = attn_output.squeeze(1)
        concat = torch.cat([x, attn_output], dim=1)
        gate = torch.sigmoid(self.gate_fc(concat))
        return gate * attn_output

#####################################
# SelfTeach Module for Reinforcement Learning of Reasoning
#####################################
class SelfTeachModule(nn.Module):
    def __init__(self, embed_dim):
        super(SelfTeachModule, self).__init__()
        self.critic = nn.Linear(embed_dim, 1)
    def forward(self, cot_embedding):
        return self.critic(cot_embedding)

#####################################
# Custom Multi-Head Latent Attention (from scratch)
#####################################
class MultiHeadLatentAttentionCustom(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim):
        super(MultiHeadLatentAttentionCustom, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.latent_proj = nn.Linear(embed_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, embed_dim)

    def forward(self, x):
        B, T, _ = x.size()
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.latent_proj(self.W_k(x)).view(B, T, self.num_heads, self.latent_dim//self.num_heads).transpose(1,2)
        V = self.latent_proj(self.W_v(x)).view(B, T, self.num_heads, self.latent_dim//self.num_heads).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, T, self.latent_dim)
        return self.out_proj(out)

#####################################
# Advanced Chain-of-Thought Generator
#####################################
class ChainOfThoughtGeneratorAdvanced(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim,
                 max_len=256, num_samples=5):
        super(ChainOfThoughtGeneratorAdvanced, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len
        self.num_samples = num_samples

    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt_ids)
        x = apply_rotary_positional_encoding(x)
        out = self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.fc_out(out)

    def generate_multiple(self, prompt_ids):
        B, L = prompt_ids.size()
        all_gen, all_logp = [], []
        for _ in range(self.num_samples):
            gen = prompt_ids.clone()
            logp = torch.zeros(B,1, device=prompt_ids.device)
            mem = apply_rotary_positional_encoding(self.embed(prompt_ids))
            for _ in range(self.max_len - L):
                logits = self.forward(gen, mem)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                nxt = torch.multinomial(probs, 1)
                logp += torch.log(probs.gather(1, nxt) + 1e-8)
                gen = torch.cat([gen, nxt], dim=1)
                if (nxt==0).all(): break
            all_gen.append(gen); all_logp.append(logp)
        return torch.stack(all_gen), torch.stack(all_logp)

    def generate_with_prompt(self, prompt_ids):
        return self.generate_multiple(prompt_ids)

#####################################
# Standard Image, Audio, Video Encoder/Decoder Modules (as before)
#####################################
class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 8 * 8, out_dim)
    def forward(self, x):
        B = x.size(0)
        feat = self.conv(x)
        feat = feat.view(B, -1)
        return self.fc(feat)

class ImageDecoder(nn.Module):
    def __init__(self, in_dim):
        super(ImageDecoder, self).__init__()
        self.fc = nn.Linear(in_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, 128, 8, 8)
        return self.deconv(x)

class AudioEncoder(nn.Module):
    def __init__(self, out_dim):
        super(AudioEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, 15, stride=4, padding=7),
            nn.ReLU()
        )
        self.fc = nn.Linear(32 * 1000, out_dim)
    def forward(self, x):
        B = x.size(0)
        feat = self.conv(x)
        feat = feat.view(B, -1)
        return self.fc(feat)

class AudioDecoder(nn.Module):
    def __init__(self, in_dim, output_length):
        super(AudioDecoder, self).__init__()
        self.fc = nn.Linear(in_dim, 32 * 1000)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 15, stride=4, padding=7, output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 15, stride=4, padding=7, output_padding=3),
            nn.Tanh()
        )
        self.output_length = output_length
    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, 32, 1000)
        return self.deconv(x)

class VideoEncoder(nn.Module):
    def __init__(self, out_dim):
        super(VideoEncoder, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,4,4))
        )
        self.fc = nn.Linear(32 * 4 * 4, out_dim)
    def forward(self, x):
        B = x.size(0)
        feat = self.conv3d(x)
        feat = feat.view(B, -1)
        return self.fc(feat)

class VideoDecoder(nn.Module):
    def __init__(self, in_dim, num_frames, frame_size):
        super(VideoDecoder, self).__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.fc = nn.Linear(in_dim, 32 * num_frames * 8 * 8)
        self.deconv3d = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 3, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.Tanh()
        )
    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, 32, self.num_frames, 8, 8)
        return self.deconv3d(x)

#####################################
# Transformer-Based Text Components
#####################################
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=512):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.max_len = max_len
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = apply_rotary_positional_encoding(x)
        return self.encoder(x)

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=512):
        super(TextDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.max_len = max_len
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.kv_cache = KVCache()
    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt_ids)
        x = apply_rotary_positional_encoding(x)
        return self.fc_out(self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask))
    def generate(self, prompt_ids):
        self.kv_cache.reset()
        generated = prompt_ids
        memory = apply_rotary_positional_encoding(self.embed(prompt_ids))
        for _ in range(self.max_len - prompt_ids.size(1)):
            logits = self.forward(generated, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == 0).all():
                break
        return generated
    def generate_with_prompt(self, prompt_ids):
        return self.generate(prompt_ids)

#####################################
# Unified Multimodal Model with All Advanced Modules
#####################################
class UnifiedMultimodalModel(nn.Module):
    def __init__(self, config):
        super(UnifiedMultimodalModel, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Text branch
        self.text_encoder = TextEncoder(config['text_vocab_size'], config['text_embed_dim'],
                                        config['text_encoder_layers'], config['text_num_heads'],
                                        config['text_ff_dim'], max_len=config.get('text_max_len',512))
        self.text_decoder = TextDecoder(config['text_vocab_size'], config['text_embed_dim'],
                                        config['text_decoder_layers'], config['text_num_heads'],
                                        config['text_ff_dim'], max_len=config.get('text_max_len',512))
        # Audio branch
        self.audio_encoder = AudioEncoder(config['audio_latent_dim'])
        self.audio_decoder = AudioDecoder(config['audio_latent_dim'], config['audio_output_length'])
        # Image branch
        self.image_encoder = ImageEncoder(config['image_latent_dim'])
        self.image_decoder = ImageDecoder(config['image_latent_dim'])
        # Video branch
        self.video_encoder = VideoEncoder(config['video_latent_dim'])
        self.video_decoder = VideoDecoder(config['video_latent_dim'], config['video_num_frames'], config['video_frame_size'])
        # Diffusion module for image refinement
        self.diffusion_module = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.diffusion_proj = nn.Linear(3 * 64 * 64, config['image_latent_dim'])
        # Combine embeddings from all encoders
        self.combined_projection = nn.Linear(
            config['text_embed_dim'] + config['audio_latent_dim'] + config['image_latent_dim'] + config['video_latent_dim'],
            config['text_embed_dim']
        )
        # Fusion modules
        input_dims = {
            "text": config['text_embed_dim'],
            "audio": config['audio_latent_dim'],
            "image": config['image_latent_dim'],
            "video": config['video_latent_dim']
        }
        self.core_fusion = CoreFusion(input_dims, config['core_fused_dim'])
        self.external_fusion = ExternalFusion(input_dims, config['external_fused_dim'])
        # Replace standard multihead attention with our custom latent attention module
        self.latent_attention = MultiHeadLatentAttentionCustom(config['external_fused_dim'], config['attention_num_heads'], latent_dim=128)
        # Advanced modules
        self.titans_memory = TitansMemoryMAC(memory_size=256, embedding_dim=config['text_embed_dim'])
        self.surprise_and_forget = SurpriseAndForget(embedding_dim=config['text_embed_dim'], threshold=0.5)
        self.deepseeks_reasoning = DeepseeksReasoning(input_dim=config['text_embed_dim'], hidden_dim=256, num_layers=2)
        # Advanced Chain-of-Thought Generator with multiple sampling
        self.cot_generator = ChainOfThoughtGeneratorAdvanced(config['text_vocab_size'], config['text_embed_dim'],
                                                               config['cot_decoder_layers'], config['text_num_heads'],
                                                               config['text_ff_dim'], max_len=config.get('cot_max_len',256),
                                                               num_samples=5)
        # SelfTeach module for reinforcement learning-based reward shaping
        self.selfteach = SelfTeachModule(config['text_embed_dim'])
        # Retrieval Component
        self.retriever = Retriever(config.get('rag_documents', ["Default document content."]))
        self.rag_generator = RAGGenerator(self.cot_generator, self.retriever)
        # Function caller
        self.func_caller = FunctionCaller()
    
    def forward(self, inputs):
        outputs = {}
        branch_features = {}
        # Process text branch
        if "text" in inputs:
            text_enc = self.text_encoder(inputs["text"])
            branch_features["text"] = text_enc[:, 0, :]
            memory_text = text_enc.mean(dim=1, keepdim=True)
            outputs["text_out"] = self.text_decoder(inputs["text"], memory_text)
        # Process audio branch
        if "audio" in inputs:
            branch_features["audio"] = self.audio_encoder(inputs["audio"])
            outputs["audio_out"] = self.audio_decoder(branch_features["audio"])
        # Process image branch
        if "image" in inputs:
            img_feat = self.image_encoder(inputs["image"])
            diffused = self.diffusion_module(inputs["image"])
            diffused_flat = torch.flatten(diffused, 1)
            diffused_proj = self.diffusion_proj(diffused_flat)
            branch_features["image"] = (img_feat + diffused_proj) / 2
            outputs["image_out"] = self.image_decoder(branch_features["image"])
        # Process video branch
        if "video" in inputs:
            branch_features["video"] = self.video_encoder(inputs["video"])
            outputs["video_out"] = self.video_decoder(branch_features["video"])
        # Combine embeddings from all branches
        combined = torch.cat([branch_features["text"], branch_features["audio"], branch_features["image"], branch_features["video"]], dim=1)
        fused_embedding = self.combined_projection(combined)
        outputs["combined_embedding"] = fused_embedding
        # Apply Titan's Memory MAC with gating and then Surprise and Forget
        memory_out = self.titans_memory(fused_embedding)
        sf_embedding = self.surprise_and_forget(memory_out)
        outputs["memory_filtered"] = sf_embedding
        # Fusion modules
        core_fused = self.core_fusion(branch_features)
        ext_fused = self.external_fusion(branch_features)
        outputs["core_fused"] = core_fused
        outputs["external_fused"] = ext_fused
        attn_input = ext_fused.unsqueeze(1)
        attn_output = self.latent_attention(attn_input)
        outputs["attended_fused"] = attn_output.squeeze(1)
        # If query is provided, generate multiple chain-of-thoughts and apply SelfTeach
        if "query" in inputs:
            # Generate multiple chain-of-thought paths and obtain log probabilities
            generated_ids, log_probs = self.cot_generator.generate_with_prompt(inputs["query"])
            # Compute an average embedding for each generated path
            B = inputs["query"].size(0)
            cot_embeds = []
            for i in range(self.cot_generator.num_samples):
                curr_ids = generated_ids[i]  # shape: (batch, seq_len)
                embed = self.text_encoder.embed(curr_ids).mean(dim=1)  # (batch, embed_dim)
                cot_embeds.append(embed)
            cot_embeds = torch.stack(cot_embeds, dim=0)  # (num_samples, batch, embed_dim)
            # Compute reward for each path using SelfTeach module
            rewards = self.selfteach(cot_embeds.mean(dim=0))  # (batch, 1)
            # For simplicity, select the highest rewarded path (this is a surrogate for full RL)
            best_path_idx = torch.argmax(rewards, dim=0)  # (batch,)
            best_paths = []
            for b in range(B):
                best_paths.append(generated_ids[best_path_idx[b], b])
            best_paths = torch.stack(best_paths, dim=0)  # (batch, seq_len)
            # Refine the best chain-of-thought using Deepseeks Reasoning
            cot_embed = self.text_encoder.embed(best_paths).mean(dim=1)  # (batch, embed_dim)
            refined_cot = self.deepseeks_reasoning(cot_embed)
            outputs["cot_out"] = refined_cot
            # Also produce RAG output (for retrieval-augmented generation)
            outputs["rag_out"] = self.rag_generator(inputs["query"])
            # Optionally, output self-teach loss (here computed as negative reward * log probability)
            outputs["selfteach_loss"] = - (log_probs.mean() * rewards.mean())
        return outputs

    def call_function(self, func_name, *args, **kwargs):
        return self.func_caller.call(func_name, *args, **kwargs)

if __name__ == "__main__":
    config = {
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
        "image_size": (3, 64, 64),
        "training_datasets": []  # will be set in train.py
    }
    model = UnifiedMultimodalModel(config)
    print("UnifiedMultimodalModel instantiated successfully.")
