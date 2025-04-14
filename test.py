#!/usr/bin/env python3
"""
test.py

Interactive Chat Loop for TitanModel

This script loads the TitanModel (from model.py) and the vocabulary (from vocab.json),
then enters a chat loop. The user can type a query and receive a chain-of-thought AI-generated
response. The conversation continues until the user types "exit", "quit", or "goodbye".

The script loads the model fully on the GPU and sets a per-process memory fraction so that
90% of the GPU memory is available for the model/inference, leaving 10% in reserve for peak usage.

Suitable for deployment on RunPod H200, with GPU acceleration enabled.
"""

import torch
import json
import os
from model import TitanModel

# Define VOCAB_SIZE as used during training.
VOCAB_SIZE = 5000

def load_vocab(vocab_path="vocab.json"):
    """Load vocabulary from a JSON file."""
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab

def tokenize_text(text, vocab, max_length=128):
    """Simple whitespace tokenizer to convert text into token IDs."""
    tokens = text.split()
    token_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in tokens][:max_length]
    if len(token_ids) < max_length:
        token_ids += [vocab.get("<pad>", 0)] * (max_length - len(token_ids))
    return token_ids

def decode_tokens(token_ids, rev_vocab):
    """Convert token IDs back into a readable string."""
    tokens = [rev_vocab.get(str(tok), "<unk>") for tok in token_ids if tok != 0]
    return " ".join(tokens)

def main():
    # Ensure the vocabulary file exists
    if not os.path.exists("vocab.json"):
        print("No vocab.json found; please run train.py first to generate the vocabulary.")
        return

    # Load vocabulary and build a reverse lookup dictionary
    vocab = load_vocab("vocab.json")
    rev_vocab = {str(idx): word for word, idx in vocab.items()}
    print("Vocabulary loaded from vocab.json.")

    # Define the model configuration (must match training settings)
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

    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If using GPU, set the per-process memory fraction to reserve a headroom (e.g. 10% reserved)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, device=device)
        torch.cuda.empty_cache()
        print("GPU memory limited to 90% of total (10% reserved for peak usage).")

    # Load the TitanModel fully onto the GPU
    model = TitanModel(config).to(device)
    model.eval()

    print("\nWelcome to TitanModel Interactive Chat!")
    print("Type your query and press Enter (type 'exit', 'quit', or 'goodbye' to end the chat).\n")

    # Main chat loop
    while True:
        user_input = input("You: ").strip()
        # End chat if user types any goodbye keyword
        if user_input.lower() in ["exit", "quit", "goodbye"]:
            print("Exiting chat. Goodbye!")
            break

        # Tokenize the input text
        token_ids = tokenize_text(user_input, vocab, max_length=50)
        # Convert to tensor (with non_blocking to speed up GPU transfers) and add batch dimension
        text_tokens = torch.tensor([token_ids], dtype=torch.long).to(device, non_blocking=True)

        # Generate response using the chain-of-thought generator in the model
        with torch.no_grad():
            generated_ids = model.cot_generator.generate_with_prompt(text_tokens)
        generated_ids = generated_ids[0].cpu().tolist()
        response = decode_tokens(generated_ids, rev_vocab)

        print("TitanModel:", response)
        print()

if __name__ == "__main__":
    main()
