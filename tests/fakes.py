"""Download-free stand-ins for a Hugging Face tokenizer and encoder."""

import torch
from torch import nn

VOCAB_SIZE = 64


class WordTokenizer:
    """Whitespace tokenizer with BERT-style special tokens; ids are stable per word."""

    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3
    mask_token = "[MASK]"

    def __init__(self):
        self.vocabulary = {}

    def _id(self, word):
        if word not in self.vocabulary:
            self.vocabulary[word] = 4 + len(self.vocabulary) % (VOCAB_SIZE - 4)
        return self.vocabulary[word]

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [self._id(word) for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.mix = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        context = (hidden * attention_mask[..., None]).sum(1, keepdim=True)
        return type("EncoderOutput", (), {
            "last_hidden_state": hidden + self.mix(context)
        })()
