import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len: int = 5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=0.1)
		
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


class Transformer(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, dropout):
		super(Transformer, self).__init__()
		self.multihead_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
		self.dropout1 = nn.Dropout(p=dropout)
		self.layer_norm1 = nn.LayerNorm(d_model)
		
		self.feed_forward = nn.Sequential(
			nn.Linear(d_model, d_ff),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.Linear(d_ff, d_model)
		)
		self.dropout2 = nn.Dropout(p=dropout)
		self.layer_norm2 = nn.LayerNorm(d_model)
		
	def forward(self, x):
		attn_output, _ = self.multihead_attention(x, x, x)
		x = x + self.dropout1(attn_output)
		x = self.layer_norm1(x)
		
		ff_output = self.feed_forward(x)
		x = x + self.dropout2(ff_output)
		x = self.layer_norm2(x)
		
		return x