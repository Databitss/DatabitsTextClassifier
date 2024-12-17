import torch
from torch import nn
import torch.nn.functional as F

class FASTTEXTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, embedding_dim)
        self.layers = [nn.Linear(embedding_dim, embedding_dim) for _ in range(n_layers)]
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeddings = self.embedding(x)
        avg_embeddings = embeddings.mean(dim=1).squeeze(1)
        for layer in self.layers:
            avg_embeddings = self.dropout(F.leaky_relu(self.hidden(avg_embeddings)))
        out = self.fc(avg_embeddings)
        return out
        
