import torch
import torch.nn as nn

from model.layers import (
	PositionalEncoding,
	Transformer
)


class TransformerLM(nn.Module):
	"""
	Name: TransformerLM
	Version: 0.0.1
	Architecture: Transformer-based Language Model

	Parameters:
	- n_vocab (int): Number of vocabulary
	- n_embedding (int): Size of word embeddings
	- n_heads (int): Number of attention heads
	- n_hidden (int): Number of hidden units in the feed-forward network
	- n_layers (int): Number of Transformer layers
	- dropout (float): Dropout probability
	"""
	def __init__(
		self,
		n_vocab: int,
		n_embedding: int,
		n_heads: int,
		n_hidden: int,
		n_layers: int,
		dropout: float,
		max_length: int = 5000
	):
		super(TransformerLM, self).__init__()

		self.embedding = nn.Embedding(n_vocab, n_embedding)
		self.positional_encoding = PositionalEncoding(n_embedding, max_length)
		
		self.transformer_layers = nn.ModuleList([
			Transformer(n_embedding, n_heads, n_hidden, dropout)
			for _ in range(n_layers)
		])

		self.fc = nn.Linear(n_embedding, n_vocab)

	def forward(self, inputs):
		embeddings = self.embedding(inputs)
		embeddings = self.positional_encoding(embeddings)

		x = embeddings
		for transformer_layer in self.transformer_layers:
			x = transformer_layer(x)

		logits = self.fc(x)
		return logits