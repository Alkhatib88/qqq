#!/usr/bin/env python3
"""
model.py

TitanModel: A Multimodal Encoder-Decoder with Advanced Reasoning, 
Function Call Handling, and Retrieval-Augmented Generation

This script implements:
  • Modality-specific encoders & decoders for text, image, audio, and video.
  • A FusionModule to combine latent vectors from each modality.
  • A custom Multi-Head Latent Attention module to compress the key/value stream.
  • A Chain-of-Thought (CoT) generator for reasoning.
  • A FunctionCallHandler to parse and execute simple function calls.
  • A dummy retrieval-augmented generator (RAG) to simulate retrieval.

Run this script directly (e.g. `python3 model.py`) to perform a forward pass 
and test a function call. Compatible with RunPod H200 GPU pods.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable CUDNN benchmark for GPU performance
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True

##############################
# Positional Encoding Module
##############################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

##############################
# Text Encoder and Decoder
##############################
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt_ids)
        x = self.pos_encoder(x)
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.fc_out(output)
        return logits

##############################
# Image Encoder and Decoder
##############################
class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        B = x.size(0)
        features = self.conv(x)
        features = features.view(B, -1)
        latent = self.fc(features)
        return latent

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, 256, 4, 4)
        img = self.deconv(x)
        return img

##############################
# Audio Encoder and Decoder
##############################
class AudioEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.ReLU()
        )
        self.fc = nn.Linear(64 * 250, latent_dim)

    def forward(self, x):
        B = x.size(0)
        features = self.conv(x)
        features = features.view(B, -1)
        latent = self.fc(features)
        return latent

class AudioDecoder(nn.Module):
    def __init__(self, latent_dim, output_length):
        super().__init__()
        self.output_length = output_length
        self.fc = nn.Linear(latent_dim, 64 * 250)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.Tanh()
        )

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, 64, 250)
        audio = self.deconv(x)
        return audio

##############################
# Video Encoder and Decoder
##############################
class VideoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        B = x.size(0)
        features = self.conv3d(x)
        features = features.view(B, -1)
        latent = self.fc(features)
        return latent

class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.fc = nn.Linear(latent_dim, 64*4*4*2)
        self.deconv3d = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(16, output_shape[0], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1)),
            nn.Tanh()
        )

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, 64, 2, 4, 4)
        video = self.deconv3d(x)
        return video

##############################
# Fusion Module
##############################
class FusionModule(nn.Module):
    def __init__(self, input_dims, fused_dim):
        super().__init__()
        total_dim = sum(input_dims.values())
        self.fc = nn.Linear(total_dim, fused_dim)
        self.activation = nn.ReLU()

    def forward(self, latent_dict):
        latents = [latent_dict[key] for key in sorted(latent_dict.keys())]
        concat = torch.cat(latents, dim=1)
        fused = self.activation(self.fc(concat))
        return fused

##############################
# Multi-Head Latent Attention
##############################
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_pool = nn.Linear(self.head_dim, latent_dim)
        self.k_pool = nn.Linear(self.head_dim, latent_dim)
        self.v_pool = nn.Linear(self.head_dim, latent_dim)
        self.out_proj = nn.Linear(num_heads * latent_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        B, T, _ = query.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        latent_k = self.k_pool(k.mean(dim=2))
        latent_v = self.v_pool(v.mean(dim=2))
        q_latent = self.q_pool(q)
        latent_k_exp = latent_k.unsqueeze(2)
        attn_scores = (q_latent * latent_k_exp).sum(dim=-1, keepdim=True)
        attn_weights = F.softmax(attn_scores, dim=2)
        weighted_q = (attn_weights * q_latent).sum(dim=2)
        context = latent_v + weighted_q
        context = context.view(B, -1)
        output = self.out_proj(context)
        return output

##############################
# Chain-of-Thought Generator
##############################
class ChainOfThoughtGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt_ids)
        x = self.pos_encoder(x)
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.fc_out(output)
        return logits

    def generate(self, prompt_ids):
        B = prompt_ids.size(0)
        generated = prompt_ids
        emb = self.embed(prompt_ids)
        memory = self.pos_encoder(emb)
        for _ in range(self.max_len - prompt_ids.size(1)):
            logits = self.forward(generated, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == 0).all():
                break
        return generated

    def generate_with_prompt(self, prompt_ids):
        return self.generate(prompt_ids)

##############################
# Function Call Handler
##############################
class FunctionCallHandler:
    def __init__(self):
        self.functions = {
            "add": self.add,
            "multiply": self.multiply,
            "subtract": self.subtract
        }
    def add(self, a, b):
        return a + b
    def multiply(self, a, b):
        return a * b
    def subtract(self, a, b):
        return a - b
    def handle_call(self, call_string):
        try:
            fname, args_str = call_string.split("(", 1)
            args_str = args_str.rstrip(")")
            args = [float(x.strip()) for x in args_str.split(",")]
            if fname in self.functions:
                return self.functions[fname](*args)
            else:
                return f"Function '{fname}' not defined."
        except Exception as e:
            return f"Error parsing call: {e}"

##############################
# Dummy Retriever and RAG
##############################
class DummyRetriever:
    def __init__(self, documents):
        self.documents = documents
    def retrieve(self, query):
        query = query.lower()
        retrieved = []
        for doc in self.documents:
            if any(word in doc.lower() for word in query.split()):
                retrieved.append(doc)
        if not retrieved:
            retrieved = self.documents[:1]
        return retrieved

class RAGGenerator(nn.Module):
    def __init__(self, cot_generator, documents):
        super().__init__()
        self.cot_generator = cot_generator
        self.retriever = DummyRetriever(documents)

    def generate_with_retrieval(self, query_ids):
        query_str = " ".join([str(id.item()) for id in query_ids[0]])
        retrieved_docs = self.retriever.retrieve(query_str)
        context = " ".join(retrieved_docs)
        context_tensor = query_ids.new_zeros(query_ids.size())
        prompt = torch.cat([query_ids, context_tensor], dim=1)
        gen_ids = self.cot_generator.generate(prompt)
        return gen_ids

##############################
# TitanModel: Integrated Model
##############################
class TitanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=config['text_vocab_size'],
            embed_dim=config['text_embed_dim'],
            num_layers=config['text_encoder_layers'],
            num_heads=config['text_num_heads'],
            ff_dim=config['text_ff_dim'],
            max_len=config.get('text_max_len', 512)
        )
        self.image_encoder = ImageEncoder(latent_dim=config['image_latent_dim'])
        self.audio_encoder = AudioEncoder(latent_dim=config['audio_latent_dim'])
        self.video_encoder = VideoEncoder(latent_dim=config['video_latent_dim'])
        self.text_decoder = TextDecoder(
            vocab_size=config['text_vocab_size'],
            embed_dim=config['text_embed_dim'],
            num_layers=config['text_decoder_layers'],
            num_heads=config['text_num_heads'],
            ff_dim=config['text_ff_dim'],
            max_len=config.get('text_max_len', 512)
        )
        self.image_decoder = ImageDecoder(latent_dim=config['image_latent_dim'])
        self.audio_decoder = AudioDecoder(latent_dim=config['audio_latent_dim'],
                                          output_length=config['audio_output_length'])
        self.video_decoder = VideoDecoder(latent_dim=config['video_latent_dim'],
                                          output_shape=config['video_output_shape'])
        input_dims = {
            "audio": config['audio_latent_dim'],
            "image": config['image_latent_dim'],
            "text": config['text_embed_dim'],
            "video": config['video_latent_dim']
        }
        self.fusion = FusionModule(input_dims=input_dims, fused_dim=config['fused_dim'])
        self.latent_attention = MultiHeadLatentAttention(
            embed_dim=config['fused_dim'],
            num_heads=config['attention_num_heads'],
            latent_dim=config['attention_latent_dim']
        )
        self.cot_generator = ChainOfThoughtGenerator(
            vocab_size=config['text_vocab_size'],
            embed_dim=config['text_embed_dim'],
            num_layers=config['cot_decoder_layers'],
            num_heads=config['text_num_heads'],
            ff_dim=config['text_ff_dim'],
            max_len=config.get('cot_max_len', 256)
        )
        self.function_handler = FunctionCallHandler()
        self.rag_generator = RAGGenerator(
            cot_generator=self.cot_generator,
            documents=config.get('rag_documents', ["Default document content."])
        )

    def forward(self, text_input_ids, image_input, audio_input, video_input, query_ids):
        latent_text_full = self.text_encoder(text_input_ids)
        latent_text = latent_text_full[:, 0, :]
        latent_image = self.image_encoder(image_input)
        latent_audio = self.audio_encoder(audio_input)
        latent_video = self.video_encoder(video_input)
        latent_dict = {"audio": latent_audio, "image": latent_image, "text": latent_text, "video": latent_video}
        fused_latent = self.fusion(latent_dict)
        fused_seq = fused_latent.unsqueeze(1)
        attended_latent = self.latent_attention(fused_seq, fused_seq, fused_seq)
        B = attended_latent.size(0)
        memory = attended_latent.unsqueeze(1).repeat(1, 10, 1)
        text_logits = self.text_decoder(query_ids, memory)
        image_out = self.image_decoder(latent_image)
        audio_out = self.audio_decoder(latent_audio)
        video_out = self.video_decoder(latent_video)
        cot_output = self.cot_generator.generate_with_prompt(query_ids)
        rag_output = self.rag_generator.generate_with_retrieval(query_ids)
        return {
            "text_output_logits": text_logits,
            "image_output": image_out,
            "audio_output": audio_out,
            "video_output": video_out,
            "cot_output": cot_output,
            "rag_output": rag_output
        }

    def call_function(self, call_str):
        return self.function_handler.handle_call(call_str)

def main():
    config = {
        "text_vocab_size": 10000,
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
        "cot_max_len": 128,
        "rag_documents": [
            "Document 1: Advanced techniques in multimodal learning.",
            "Document 2: Chain-of-thought prompting and reasoning improvements.",
            "Document 3: Retrieval augmented generation in modern AI."
        ]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TitanModel(config).to(device)
    B = 1
    seq_len = 20
    text_input_ids = torch.randint(0, config['text_vocab_size'], (B, seq_len), device=device)
    image_input = torch.randn(B, 3, 64, 64, device=device)
    audio_input = torch.randn(B, 1, config['audio_output_length'], device=device)
    video_input = torch.randn(B, 3, 16, 64, 64, device=device)
    query_ids = text_input_ids

    outputs = model(text_input_ids, image_input, audio_input, video_input, query_ids)

    print("=== Titan Model Outputs ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {value}")

    call_str = "multiply(4, 3)"
    func_result = model.call_function(call_str)
    print(f"\nFunction Call '{call_str}' Output: {func_result}")

if __name__ == "__main__":
    main()
