import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        hidden_state = hidden[-1]
        out = self.fc(self.dropout(hidden_state))
        return out
