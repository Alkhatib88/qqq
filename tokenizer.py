#!/usr/bin/env python3
"""
tokenizer.py

A custom tokenizer built from scratch. It builds its vocabulary from provided texts,
implements tokenization (encoding) and detokenization (decoding). Special tokens supported:
   <PAD>, <UNK>, <BOS>, <EOS>, <IMG>, <AUD>, <VID>
This module is used at both training and inference.
"""

import re

class SimpleTokenizer:
    def __init__(self, max_vocab_size=None):
        self.max_vocab_size = max_vocab_size
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<IMG>", "<AUD>", "<VID>"]
        self.token_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.special_tokens)}
        self.vocab_size = len(self.special_tokens)
        self.fitted = False

    def fit_on_texts(self, texts):
        """Build vocabulary from an iterable of text strings."""
        word_freq = {}
        for text in texts:
            text = text.lower()
            # Use regex to capture only word characters
            tokens = re.findall(r'\w+', text)
            for token in tokens:
                if token in self.token_to_id:
                    continue
                word_freq[token] = word_freq.get(token, 0) + 1
        sorted_tokens = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        if self.max_vocab_size:
            sorted_tokens = sorted_tokens[:self.max_vocab_size - len(self.special_tokens)]
        for token, freq in sorted_tokens:
            if token in self.token_to_id:
                continue
            self.token_to_id[token] = self.vocab_size
            self.id_to_token[self.vocab_size] = token
            self.vocab_size += 1
        self.fitted = True

    def tokenize(self, text):
        """Encode text into a list of token IDs."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        token_ids = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
        return token_ids

    def detokenize(self, token_ids):
        """Decode a list of token IDs back into text."""
        tokens = [self.id_to_token.get(tid, "<UNK>") for tid in token_ids if self.id_to_token.get(tid, None) not in ("<PAD>", "<BOS>", "<EOS>")]
        return " ".join(tokens)

if __name__ == "__main__":
    sample_texts = [
        "A cat sat on the mat.",
        "The dog sat on the log.",
        "An image <IMG> is represented."
    ]
    tokenizer = SimpleTokenizer(max_vocab_size=100)
    tokenizer.fit_on_texts(sample_texts)
    sample = "The cat and the dog sat together."
    ids = tokenizer.tokenize(sample)
    print("Encoded:", ids)
    print("Decoded:", tokenizer.detokenize(ids))
