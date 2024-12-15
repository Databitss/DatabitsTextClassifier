import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(sequence_length, dtype=torch.float).unsqueeze(1) 
        i = torch.arange(embed_dim, dtype=torch.float)

        angle_rates = 1 / torch.pow(10000.0, (2 * (i // 2)) / embed_dim)
        angle_rads = position * angle_rates

        pos_encoding = torch.zeros(sequence_length, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2]) 
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])  

        if embed_dim % 2 != 0:
            extra_sin = torch.sin(position * (1 / torch.pow(10000.0, embed_dim - 1)))
            pos_encoding = torch.cat([pos_encoding, extra_sin.unsqueeze(-1)], dim=-1)

        self.register_buffer('positional_encoding', pos_encoding.unsqueeze(0))

    def forward(self, x):
        seq_length = x.size(1)
        return x + self.positional_encoding[:, :seq_length, :]  

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)  
        self.W_k = nn.Linear(d_model, d_model)  
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) 
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        device = Q.device
        self.to(device)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=embed_dim, num_heads=num_heads, dropout=0.1)
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.densel = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        device = query.device
        self.to(device)
        mask = torch.ones(query.size(0), 1, query.size(1), key.size(1)).to(device)
        x_ = self.attention(query, key, value, mask=mask)
        x = self.layernorm_1(query+x_)
        x_ = self.densel(self.dropout(self.dense(x)))
        return self.layernorm_2(x+x_)
