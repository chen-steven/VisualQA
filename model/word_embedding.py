import torch
import torch.nn as nn
import os
import pickle 

class WordEmbedding(nn.Module):
	def __init__(self, path, pad_idx):
		super(WordEmbedding, self).__init__()
		weights = self.load_weights(path)
		weights = torch.tensor(weights)
		n, embed_size = weights.size()
		self.embedding = nn.Embedding(n, embed_size, padding_idx=pad_idx)
		self.embedding.weight = nn.Parameter(weights)
		self.embedding.weight.requires_grad = False

	def forward(self, input):
		return self.embedding(input)

	def load_weights(self, path):
		data = None
		with open(path, 'rb') as f:
			data = pickle.load(f)
		return data