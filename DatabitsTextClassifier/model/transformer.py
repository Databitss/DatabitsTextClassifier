import torch
import torch.nn as nn
from .module import *

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, sequence_length=64, 
                 vocab_size=10000, n_layers=2, bert=False, num_cls=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(sequence_length=sequence_length + 1, embed_dim=embed_dim)
        self.attention = [TransformerBlock(embed_dim, num_heads) for _ in range(n_layers)]
        self.fc = nn.Linear(embed_dim , embed_dim)
        self.out = nn.Linear(embed_dim, num_cls)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs = self.embedding(inputs)
        inputs = self.positional_encoding(inputs)

        for layer in self.attention:
            inputs = layer(inputs, inputs, inputs)
          
        cls = torch.mean(inputs, dim=1)
        out = self.out(self.fc(cls))
        return out
