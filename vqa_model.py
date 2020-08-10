import torch.nn as nn
import torch
from model.word_embedding import WordEmbedding
from model.top_down_attention import TopDownAttention
from model.mlp import MLP
from model.classifier import Classifier
from model.encoder import Encoder

class VQAModel(nn.Module):
	def __init__(self, num_embeddings, embed_dim, v_dim = 2048, q_dim=1024, common_dim=512, a_dim=3133):
		super(VQAModel, self).__init__()
		self.v_dim = v_dim
		self.embeddings = nn.Embedding(num_embeddings, embed_dim, padding_idx=0)
		# self.embeddings = WordEmbedding('data', 0)
		self.encoder = Encoder(embed_dim, q_dim, 2, 0.2)
		self.top_down_att = TopDownAttention(v_dim, q_dim, common_dim)
		# self.v_proj = MLP(v_dim,common_dim)
		# self.q_proj = MLP(q_dim, common_dim)

		self.v_proj = nn.Linear(v_dim, common_dim, bias=True)
		self.q_proj = nn.Linear(q_dim, common_dim, bias=True)

		self.classifier = Classifier(common_dim, a_dim)

	def forward(self, question, q_lens, features, n_objs):
		embeddings = self.embeddings(question)
		cell, hidden,outputs = self.encoder(embeddings, q_lens)
		a = self.top_down_att(features, cell, n_objs)
		a = a.repeat(1,1,self.v_dim)

		v = torch.sum(a*features, 1)

		v = self.v_proj(v)
		q = self.q_proj(cell)

		joint_repr = v*q
		pred = self.classifier(joint_repr)
		return pred

	def grad_params(self):
		for p in self.parameters():
			if p.requires_grad:
				yield p

if __name__ == '__main__':
	from dataset import Dataset
	dataset = Dataset('', 'train')
	# print(dataset.word2ind)
	model = VQAModel(18248, 300)
	f, bb, spat, obs, a, q, q_len, qid = dataset[0]
	test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
	# print(model.forward(q, q_len, f, obs))
	for f, bb, spat, obs, a, q, q_len, qid in test_loader:

		# print(torch.narrow(f, 1, 0, obs))
		v = model.forward(q,q_len, f,obs)
		print(v.size())
		break
