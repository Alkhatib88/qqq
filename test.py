# test.py

#!/usr/bin/env python3
"""
test.py

Interactive loop for UnifiedMultimodalModel.  
Supports simple function‐call commands.
"""

import torch
from model import UnifiedMultimodalModel
from tokenizer import SimpleTokenizer

def get_default_config():
    # must match train.py’s config
    return {
      "text_vocab_size":10000,"text_embed_dim":512,
      "text_encoder_layers":2,"text_decoder_layers":2,
      "text_num_heads":8,"text_ff_dim":1024,"text_max_len":128,
      "cot_decoder_layers":2,"cot_max_len":128,
      "audio_latent_dim":256,"audio_output_length":16000,
      "image_latent_dim":256,"video_latent_dim":256,
      "video_num_frames":16,"video_frame_size":(64,64),
      "core_fused_dim":512,"external_fused_dim":512,
      "attention_num_heads":8,
      "rag_documents":["Document 1","Document 2","Document 3"]
    }

def load_model(cfg, path="unified_model.pt"):
    model = UnifiedMultimodalModel(cfg)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        st = torch.load(path, map_location=dev)
        model.load_state_dict(st)
        print(f"Loaded weights from {path}")
    except Exception as e:
        print(f"Could not load {path}, using random init ({e})")
    model.to(dev).eval()
    return model

def interactive_loop(model, tokenizer, cfg):
    print("Type a message; 'bye' to exit.")
    while True:
        msg = input("You: ").strip()
        if msg.lower()=="bye":
            break
        if msg.startswith("build_script:"):
            out = model.call_function("build_script", msg.split(":",1)[1].strip())
            print("Model:", out); continue
        if msg.startswith("execute_script:"):
            out = model.call_function("execute_script", msg.split(":",1)[1].strip())
            print("Model:", out); continue

        ids = tokenizer.tokenize(msg)
        t   = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
        dummy_audio = torch.zeros(1, cfg["audio_output_length"]).to(model.device)
        dummy_image = torch.zeros(1,3,64,64).to(model.device)
        dummy_video = torch.zeros(1,3,cfg["video_num_frames"],64,64).to(model.device)

        inp = {"text":t, "query":t, "audio":dummy_audio, "image":dummy_image, "video":dummy_video}
        with torch.no_grad():
            out = model(inp)
            resp = out.get("cot_out", t).cpu().numpy()[0]
        print("Model:", tokenizer.detokenize(resp))

def main():
    cfg = get_default_config()
    tk  = SimpleTokenizer(max_vocab_size=cfg["text_vocab_size"])
    tk.fit_on_texts(["init vocabulary"])
    mdl = load_model(cfg)
    interactive_loop(mdl, tk, cfg)

if __name__=="__main__":
    main()
