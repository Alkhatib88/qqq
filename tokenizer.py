# tokenizer.py

#!/usr/bin/env python3
"""
tokenizer.py

A simple regex‐based tokenizer with fit/detokenize.
Special tokens: <PAD>,<UNK>,<BOS>,<EOS>,<IMG>,<AUD>,<VID>
"""

import re

class SimpleTokenizer:
    def __init__(self, max_vocab_size=None):
        self.max_vocab_size = max_vocab_size
        self.special_tokens = ["<PAD>","<UNK>","<BOS>","<EOS>","<IMG>","<AUD>","<VID>"]
        self.token_to_id = {tok:i for i,tok in enumerate(self.special_tokens)}
        self.id_to_token = {i:tok for i,tok in enumerate(self.special_tokens)}
        self.vocab_size  = len(self.special_tokens)
        self.fitted      = False

    def fit_on_texts(self, texts):
        freq = {}
        for text in texts:
            for tok in re.findall(r'\w+', text.lower()):
                if tok in self.token_to_id: continue
                freq[tok] = freq.get(tok,0)+1
        sorted_ = sorted(freq.items(), key=lambda x:-x[1])
        limit = self.max_vocab_size - self.vocab_size if self.max_vocab_size else None
        for tok,_ in sorted_[:limit]:
            self.token_to_id[tok] = self.vocab_size
            self.id_to_token[self.vocab_size] = tok
            self.vocab_size += 1
        self.fitted = True

    def tokenize(self, text):
        return [ self.token_to_id.get(tok, self.token_to_id["<UNK>"])
                 for tok in re.findall(r'\w+', text.lower()) ]

    def detokenize(self, token_ids):
        toks = [ self.id_to_token.get(i,"<UNK>")
                 for i in token_ids
                 if self.id_to_token.get(i) not in ("<PAD>","<BOS>","<EOS>")]
        return " ".join(toks)

if __name__=="__main__":
    tk = SimpleTokenizer(max_vocab_size=100)
    tk.fit_on_texts(["Hello world","Hello ChatGPT"])
    ids = tk.tokenize("Hello unknown token")
    print("IDs:", ids)
    print("Decoded:", tk.detokenize(ids))
