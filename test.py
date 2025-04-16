#!/usr/bin/env python3
"""
test.py

Interactive test script for the UnifiedMultimodalModel.
Loads the trained model and tokenizer, then enters an interactive chat loop.
Commands for function calls are supported.
Type 'bye' to exit.
"""

import torch
from model import UnifiedMultimodalModel, get_default_config
from tokenizer import SimpleTokenizer

def load_model(config, model_path="unified_model.pt"):
    model = UnifiedMultimodalModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded trained model weights from {model_path}.")
    except Exception as e:
        print(f"Failed to load weights from {model_path}. Using randomly initialized model. ({e})")
    model.to(device)
    model.eval()
    return model

def interactive_loop(model, tokenizer, config):
    seq_len = config.get("text_seq_len", 32)
    print("Interactive mode. Type your message.\nCommands:\n  'build_script: <script_name>'\n  'execute_script: <script_name>'\nType 'bye' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "bye":
            print("Shutting down. Goodbye!")
            break
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
        token_ids = tokenizer.tokenize(user_input)
        text_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(model.device)
        dummy_audio = torch.zeros(1, config["audio_output_length"]).to(model.device)
        dummy_image = torch.zeros(1, 3, config.get("image_size", (224,224))[1], config.get("image_size", (224,224))[2]).to(model.device)
        dummy_video = torch.zeros(1, 3, config.get("video_num_frames", 16), config.get("video_frame_size", (64,64))[0], config.get("video_frame_size", (64,64))[1]).to(model.device)
        inputs = {
            "text": text_tensor,
            "query": text_tensor,
            "audio": dummy_audio,
            "image": dummy_image,
            "video": dummy_video
        }
        with torch.no_grad():
            outputs = model(inputs)
            response_embedding = outputs.get("cot_out", text_tensor)
        decoded_response = tokenizer.detokenize(response_embedding.cpu().tolist()[0])
        print("Model:", decoded_response)

def main():
    config = get_default_config()
    tokenizer = SimpleTokenizer(max_vocab_size=config["text_vocab_size"])
    sample_texts = ["this is an example to build the tokenizer vocabulary"]
    tokenizer.fit_on_texts(sample_texts)
    model = load_model(config)
    interactive_loop(model, tokenizer, config)

if __name__ == "__main__":
    main()
