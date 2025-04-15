#!/usr/bin/env python3
"""
test.py

Interactive test script for the UnifiedMultimodalModel.
This script loads the trained model (if available) and enters an interactive loop
where the user can send text messages and also issue special commands to create/execute scripts.
Typing "bye" terminates the session.
"""

import torch
from model import UnifiedMultimodalModel, get_default_config
import sys

VOCAB_SIZE = 10000

def tokenize(text, seq_len=32):
    words = text.strip().split()
    tokens = [ord(w[0]) % VOCAB_SIZE for w in words if w]
    if len(tokens) < seq_len:
        tokens += [0]*(seq_len - len(tokens))
    else:
        tokens = tokens[:seq_len]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

def detokenize(token_tensor):
    token_list = token_tensor.squeeze(0).tolist()
    return "".join([chr((int(t) % 26) + 97) for t in token_list])

def load_model(config, model_path="unified_model.pt"):
    model = UnifiedMultimodalModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded trained model weights from {model_path}.")
    except Exception as e:
        print(f"Failed to load weights from {model_path}. Using random initialized model. ({e})")
    model.eval()
    return model

def interactive_loop(model, config):
    seq_len = config.get("text_seq_len", 32)
    print("Interactive mode. Type your message. Use commands:\n" +
          "  'build_script: <script_name>'\n" +
          "  'execute_script: <script_name>'\n" +
          "Type 'bye' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "bye":
            print("Shutting down. Goodbye!")
            break
        # Check for special commands
        if user_input.startswith("build_script:"):
            script_name = user_input.split("build_script:", 1)[1].strip()
            result = model.call_function("build_script", script_name)
            print("Model:", result)
            continue
        if user_input.startswith("execute_script:"):
            script_name = user_input.split("execute_script:", 1)[1].strip()
            result = model.call_function("execute_script", script_name)
            print("Model:", result)
            continue
        # Otherwise, process as normal text query
        text_tensor = tokenize(user_input, seq_len=seq_len).to(model.device)
        dummy_audio = torch.zeros(1, config["audio_output_length"]).to(model.device)
        dummy_image = torch.zeros(1, 3, config.get("image_size", (3,64,64))[1], config.get("image_size", (3,64,64))[2]).to(model.device)
        dummy_video = torch.zeros(1, 3, config.get("video_num_frames",16), config.get("video_frame_size", (64,64))[0], config.get("video_frame_size", (64,64))[1]).to(model.device)
        inputs = {
            "text": text_tensor,
            "query": text_tensor,
            "audio": dummy_audio,
            "image": dummy_image,
            "video": dummy_video
        }
        with torch.no_grad():
            outputs = model(inputs)
            response_ids = outputs.get("cot_out", text_tensor)
        response = detokenize(response_ids[0].cpu())
        print("Model:", response)

def main():
    config = get_default_config()
    model = load_model(config)
    interactive_loop(model, config)

if __name__ == "__main__":
    main()
