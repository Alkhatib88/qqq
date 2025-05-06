# model.py

#!/usr/bin/env python3
"""
model.py

UnifiedMultimodalModel: A fully integrated multimodal encoder-decoder with all advanced modules:
  - Rotary Positional Encoding
  - Custom KV Cache
  - TitansMemoryMAC with gating
  - SelfTeachModule for RL self-teaching
  - Advanced Chain‐of‐Thought Generator
  - Custom Multi‐Head Latent Attention
  - Deepseeks Reasoning MLP
  - Standard image, audio, video encoders/decoders
  - Fusion modules (Core & External)
  - Retrieval‐Augmented Generation stub
  - Function Call Interface stub
"""

import math, torch
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
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

#####################################
# RMSNorm
#####################################
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps
        return self.weight * (x / norm)

#####################################
# KV Cache
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
# Fusion & Advanced Modules
#####################################
class CoreFusion(nn.Module):
    def __init__(self, input_dims, fused_dim):
        super().__init__()
        total = sum(input_dims.values())
        self.net = nn.Sequential(
            nn.Linear(total, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim)
        )
    def forward(self, features):
        x = torch.cat(list(features.values()), dim=-1)
        return self.net(x)

class ExternalFusion(nn.Module):
    def __init__(self, input_dims, fused_dim):
        super().__init__()
        total = sum(input_dims.values())
        self.net = nn.Sequential(
            nn.Linear(total, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim)
        )
    def forward(self, features):
        x = torch.cat(list(features.values()), dim=-1)
        return self.net(x)

class SurpriseAndForget(nn.Module):
    def __init__(self, embedding_dim, threshold=0.5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        g = self.gate(x)
        return x * g

class DeepseeksReasoning(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Retriever:
    def __init__(self, documents):
        self.documents = documents or []
    def retrieve(self, query_ids):
        # naive: always return first doc
        return [self.documents[0]] if self.documents else []

class RAGGenerator:
    def __init__(self, cot_gen, retriever):
        self.cot_gen = cot_gen
        self.retriever = retriever
    def __call__(self, query_ids):
        docs = self.retriever.retrieve(query_ids)
        # naive: just return retrieved docs list
        return docs

class FunctionCaller:
    def __init__(self):
        pass
    def call(self, name, *args, **kwargs):
        return f"Function {name} called with {args} {kwargs}"

#####################################
# TitansMemoryMAC with Gating
#####################################
class TitansMemoryMAC(nn.Module):
    def __init__(self, memory_size, embedding_dim, num_heads=4):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.gate_fc = nn.Linear(embedding_dim * 2, embedding_dim)
    def forward(self, x):
        query = x.unsqueeze(1)                              # (B,1,dim)
        key   = self.memory.unsqueeze(0).expand(x.size(0), -1, -1)
        value = key
        attn_output, _ = self.attn(query, key, value)
        attn_output = attn_output.squeeze(1)                # (B,dim)
        gate = torch.sigmoid(self.gate_fc(torch.cat([x, attn_output], dim=-1)))
        return gate * attn_output

#####################################
# SelfTeach Module
#####################################
class SelfTeachModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.critic = nn.Linear(embed_dim, 1)
    def forward(self, cot_embedding):
        return self.critic(cot_embedding)

#####################################
# Custom Multi-Head Latent Attention
#####################################
class MultiHeadLatentAttentionCustom(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible"
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.latent_proj = nn.Linear(embed_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, embed_dim)

    def forward(self, x):
        B,T,_ = x.size()
        Q = self.W_q(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        K = self.latent_proj(self.W_k(x)).view(B,T,self.num_heads,self.latent_dim//self.num_heads).transpose(1,2)
        V = self.latent_proj(self.W_v(x)).view(B,T,self.num_heads,self.latent_dim//self.num_heads).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn  = torch.softmax(scores, dim=-1)
        out   = torch.matmul(attn, V).transpose(1,2).contiguous().view(B,T,self.latent_dim)
        return self.out_proj(out)

#####################################
# Advanced Chain-of-Thought Generator
#####################################
class ChainOfThoughtGeneratorAdvanced(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim,
                 max_len=256, num_samples=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerDecoderLayer(d_model=embed_dim,
                                           nhead=num_heads,
                                           dim_feedforward=ff_dim,
                                           batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len
        self.num_samples = num_samples

    def forward(self, tgt_ids, memory, **kwargs):
        x = apply_rotary_positional_encoding(self.embed(tgt_ids))
        return self.fc_out(self.decoder(x, memory, **kwargs))

    def generate_multiple(self, prompt_ids):
        B,L = prompt_ids.size()
        gens, logs = [], []
        for _ in range(self.num_samples):
            gen  = prompt_ids.clone()
            logp = torch.zeros(B,1, device=gen.device)
            mem  = apply_rotary_positional_encoding(self.embed(prompt_ids))
            for _ in range(self.max_len - L):
                logits = self.forward(gen, mem)
                probs  = torch.softmax(logits[:, -1, :], dim=-1)
                nxt    = torch.multinomial(probs, 1)
                logp  += torch.log(probs.gather(1,nxt)+1e-8)
                gen    = torch.cat([gen, nxt], dim=1)
                if (nxt==0).all(): break
            gens.append(gen); logs.append(logp)
        return torch.stack(gens), torch.stack(logs)

    def generate_with_prompt(self, prompt_ids):
        return self.generate_multiple(prompt_ids)

#####################################
# Standard Image, Audio, Video Modules
#####################################
class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),nn.ReLU()
        )
        self.fc   = nn.Linear(128*8*8, out_dim)
    def forward(self,x):
        B = x.size(0)
        f = self.conv(x).view(B,-1)
        return self.fc(f)

class ImageDecoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim,128*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),  nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),   nn.Tanh()
        )
    def forward(self,z):
        B = z.size(0)
        x = self.fc(z).view(B,128,8,8)
        return self.deconv(x)

class AudioEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,16,15,4,7), nn.ReLU(),
            nn.Conv1d(16,32,15,4,7),nn.ReLU()
        )
        self.fc   = nn.Linear(32*1000, out_dim)
    def forward(self,x):
        B = x.size(0)
        f = self.conv(x).view(B,-1)
        return self.fc(f)

class AudioDecoder(nn.Module):
    def __init__(self,in_dim,output_length):
        super().__init__()
        self.fc  = nn.Linear(in_dim,32*1000)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32,16,15,4,7, output_padding=3), nn.ReLU(),
            nn.ConvTranspose1d(16,1,15,4,7, output_padding=3),  nn.Tanh()
        )
    def forward(self,z):
        B= z.size(0)
        x= self.fc(z).view(B,32,1000)
        return self.deconv(x)

class VideoEncoder(nn.Module):
    def __init__(self,out_dim):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3,16,(3,4,4),(1,2,2),(1,1,1)), nn.ReLU(),
            nn.Conv3d(16,32,(3,4,4),(1,2,2),(1,1,1)), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,4,4))
        )
        self.fc = nn.Linear(32*4*4,out_dim)
    def forward(self,x):
        B= x.size(0)
        f= self.conv3d(x).view(B,-1)
        return self.fc(f)

class VideoDecoder(nn.Module):
    def __init__(self,in_dim,num_frames,frame_size):
        super().__init__()
        self.num_frames= num_frames
        self.fc = nn.Linear(in_dim,32*num_frames*8*8)
        self.deconv3d= nn.Sequential(
            nn.ConvTranspose3d(32,32,(1,4,4),(1,2,2),(0,1,1)), nn.ReLU(),
            nn.ConvTranspose3d(32,16,(1,4,4),(1,2,2),(0,1,1)), nn.ReLU(),
            nn.ConvTranspose3d(16,3,(1,4,4),(1,2,2),(0,1,1)),  nn.Tanh()
        )
    def forward(self,z):
        B= z.size(0)
        x= self.fc(z).view(B,32,self.num_frames,8,8)
        return self.deconv3d(x)

#####################################
# Transformer Text Components
#####################################
class TextEncoder(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_layers,num_heads,ff_dim,max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self,input_ids):
        x = self.embed(input_ids)
        x = apply_rotary_positional_encoding(x)
        return self.encoder(x)

class TextDecoder(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_layers,num_heads,ff_dim,max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.kv_cache = KVCache()
    def forward(self,tgt_ids,memory,tgt_mask=None,memory_mask=None):
        x = self.embed(tgt_ids)
        x = apply_rotary_positional_encoding(x)
        return self.fc_out(self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask))
    def generate(self,prompt_ids):
        self.kv_cache.reset()
        gen = prompt_ids
        mem = apply_rotary_positional_encoding(self.embed(prompt_ids))
        for _ in range(self.kv_cache and prompt_ids.size(1), self.kv_cache and self.max_len or prompt_ids.size(1)):
            logits = self.forward(gen, mem)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
            if (nxt==0).all(): break
        return gen
    def generate_with_prompt(self,prompt_ids):
        return self.generate(prompt_ids)

class UnifiedMultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ─── Branch encoders/decoders ─────────────────────────────────
        self.text_encoder  = TextEncoder(**config)
        self.text_decoder  = TextDecoder(**config)
        self.audio_encoder = AudioEncoder(config['audio_latent_dim'])
        self.audio_decoder = AudioDecoder(config['audio_latent_dim'], config['audio_output_length'])
        self.image_encoder = ImageEncoder(config['image_latent_dim'])
        self.image_decoder = ImageDecoder(config['image_latent_dim'])
        self.video_encoder = VideoEncoder(config['video_latent_dim'])
        self.video_decoder = VideoDecoder(config['video_latent_dim'], config['video_num_frames'], config['video_frame_size'])

        # ─── Diffusion refinement ─────────────────────────────────────
        self.diffusion_module = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), nn.ReLU(),
            nn.Conv2d(64,64,3,1,1), nn.ReLU()
        )
        self.diffusion_proj = nn.Linear(3*64*64, config['image_latent_dim'])

        # ─── Combine all embeddings ────────────────────────────────────
        total = config['text_embed_dim'] + config['audio_latent_dim'] \
              + config['image_latent_dim'] + config['video_latent_dim']
        self.combined_projection = nn.Linear(total, config['text_embed_dim'])

        # ─── Fusion modules ───────────────────────────────────────────
        input_dims = {
          "text": config['text_embed_dim'],
          "audio": config['audio_latent_dim'],
          "image": config['image_latent_dim'],
          "video": config['video_latent_dim']
        }
        self.core_fusion     = CoreFusion(input_dims, config['core_fused_dim'])
        self.external_fusion = ExternalFusion(input_dims, config['external_fused_dim'])
        self.latent_attention = MultiHeadLatentAttentionCustom(
            config['external_fused_dim'],
            config['attention_num_heads'],
            latent_dim=128
        )

        # ─── Advanced modules ─────────────────────────────────────────
        self.titans_memory     = TitansMemoryMAC(256, config['text_embed_dim'])
        self.surprise_and_forget = SurpriseAndForget(config['text_embed_dim'])
        self.deepseeks_reasoning = DeepseeksReasoning(config['text_embed_dim'], 256, 2)

        # ─── Chain‐of‐Thought & SelfTeach ─────────────────────────────
        self.cot_generator = ChainOfThoughtGeneratorAdvanced(
            config['text_vocab_size'], config['text_embed_dim'],
            config['cot_decoder_layers'], config['text_num_heads'],
            config['text_ff_dim'], max_len=config['cot_max_len'], num_samples=5
        )
        self.selfteach     = SelfTeachModule(config['text_embed_dim'])

        # ─── Retrieval & Function Call ────────────────────────────────
        self.retriever     = Retriever(config.get('rag_documents'))
        self.rag_generator = RAGGenerator(self.cot_generator, self.retriever)
        self.func_caller   = FunctionCaller()

    def forward(self, inputs):
        outputs, feats = {}, {}

        # text
        if "text" in inputs:
            enc = self.text_encoder(inputs["text"])
            feats["text"] = enc[:, 0, :]
            mem_text     = enc.mean(1, keepdim=True)
            outputs["text_out"] = self.text_decoder(inputs["text"], mem_text)

        # audio
        if "audio" in inputs:
            feats["audio"] = self.audio_encoder(inputs["audio"])
            outputs["audio_out"] = self.audio_decoder(feats["audio"])

        # image
        if "image" in inputs:
            img_feat = self.image_encoder(inputs["image"])
            diff    = self.diffusion_module(inputs["image"])
            diff    = diff.flatten(1)
            diff    = self.diffusion_proj(diff)
            feats["image"] = (img_feat + diff) * 0.5
            outputs["image_out"] = self.image_decoder(feats["image"])

        # video
        if "video" in inputs:
            feats["video"] = self.video_encoder(inputs["video"])
            outputs["video_out"] = self.video_decoder(feats["video"])

        # combine & project
        combined = torch.cat(list(feats.values()), dim=-1)
        fused_emb = self.combined_projection(combined)
        outputs["combined_embedding"] = fused_emb

        # memory & surprise
        mem_out = self.titans_memory(fused_emb)
        sf_out  = self.surprise_and_forget(mem_out)
        outputs["memory_filtered"] = sf_out

        # fusion & attention
        core_f = self.core_fusion(feats)
        ext_f  = self.external_fusion(feats)
        outputs["core_fused"]     = core_f
        outputs["external_fused"] = ext_f
        attn_in  = ext_f.unsqueeze(1)
        outputs["attended_fused"] = self.latent_attention(attn_in).squeeze(1)

        # chain‐of‐thought + self‐teach + rag
        if "query" in inputs:
            gens, logp = self.cot_generator.generate_with_prompt(inputs["query"])
            B = inputs["query"].size(0)
            # average‐embedding of each sample
            emb_samples = []
            for i in range(self.cot_generator.num_samples):
                ids = gens[i]
                emb = self.text_encoder.embed(ids).mean(1)
                emb_samples.append(emb)
            emb_stack = torch.stack(emb_samples, 0)       # (num_samples, B, D)
            rewards   = self.selfteach(emb_stack.mean(0)) # (B,1)
            best_idx  = rewards.argmax(0)                 # (B,)
            best_paths = torch.stack([
                gens[best_idx[b]][b] for b in range(B)
            ], 0)
            refined = self.deepseeks_reasoning(
                self.text_encoder.embed(best_paths).mean(1)
            )
            outputs["cot_out"]     = refined
            outputs["rag_out"]     = self.rag_generator(inputs["query"])
            outputs["selfteach_loss"] = - (logp.mean() * rewards.mean())

        return outputs

if __name__ == "__main__":
    # simple instantiation test
    cfg = {
        "text_vocab_size": 10000, "text_embed_dim":512,
        "text_encoder_layers":2,"text_decoder_layers":2,
        "text_num_heads":8,"text_ff_dim":1024,"text_max_len":128,
        "cot_decoder_layers":2,"cot_max_len":128,
        "audio_latent_dim":256,"audio_output_length":16000,
        "image_latent_dim":256,"video_latent_dim":256,
        "video_num_frames":16,"video_frame_size":(64,64),
        "core_fused_dim":512,"external_fused_dim":512,
        "attention_num_heads":8,
        "rag_documents":["doc1","doc2","doc3"]
    }
    model = UnifiedMultimodalModel(cfg)
    print("UnifiedMultimodalModel instantiated successfully.")
