import torch
import torch.utils.data
import numpy as np
import _pickle as cPickle
import torch.nn as nn
import h5py

class Dataset(torch.utils.data.Dataset):
	def __init__(self, root_dir, mode):
		super(Dataset, self).__init__()
		self.quid = self.open('./data/dicts/ind2qid_{}.pkl'.format(mode))
		self.quid2ques = self.open('./data/dicts/qid2que_{}.pkl'.format(mode))
		self.quid2image = self.open('./data/dicts/qid2vid_{}.pkl'.format(mode))
		self.f_path = './data/adaptive_features/{}.h5'.format(mode)
		self.img2f = self.open('./data/adaptive_features/{}_img2idx.pkl'.format(mode))
		self.qid2a_inds = self.open('./data/dicts/qid2_a_inds.pkl')
		self.ind2ans = self.open('./data/dicts/ans2ind.pkl')
		self.word2ind = self.open('./data/dicts/word2ind.pkl')

		self.num_classes = len(self.ind2ans.values())
		# self.images =
	def __getitem__(self, item):
		quid = self.quid[item]
		q, q_len = self.quid2ques[quid]
		q = torch.from_numpy(np.array(q)).long()
		image_id = self.quid2image[quid]
		f, bb, spat, obs = self.load_image(image_id)
		a = self.multi_hot_encoding(quid)
		return f, bb, spat, obs, a, q, q_len, torch.tensor(int(quid))
	def __len__(self):
		return len(self.quid)
	def multi_hot_encoding(self, quid):
		inds_scores = self.qid2a_inds.get(str(quid))
		inds = inds_scores[0]
		scores = inds_scores[1]
		vec = torch.zeros(self.num_classes)
		for ind in inds:
			vec[ind] = scores[ind]
		return vec
	def load_image(self, image_id):
		hf = h5py.File(self.f_path, 'r')
		features = hf.get('image_features')
		spatials = hf.get('spatial_features')
		bb = hf.get('image_bb')
		index = self.img2f[int(image_id)]
		pos_boxes = hf.get('pos_boxes')
		last_obj = pos_boxes[index][1]
		first_obj = pos_boxes[index][0]
		bb = bb[first_obj:last_obj, :]
		bb = torch.from_numpy(bb)
		features = features[first_obj:last_obj, :]
		features = torch.from_numpy(features)
		spatials = spatials[first_obj:last_obj, :]
		spatials = torch.from_numpy(spatials)
		max_obs = 100
		obs = bb.size(0)
		pad_amount = max_obs - obs
		f = torch.cat((features, torch.zeros(pad_amount, features.size(1))))
		bb = torch.cat((bb, torch.zeros(pad_amount, bb.size(1))))
		obs = torch.Tensor([obs])
		spat = torch.cat((spatials, torch.zeros(pad_amount, spatials.size(1))))
		return f, bb, spat, obs
	def open(self, path):
		with open(path, 'rb') as fd:
			data = cPickle.load(fd)
		return data

if __name__ == '__main__':
	dataset = Dataset('', 'train')
	f, bb, spat, obs, a, q, q_len,qid = dataset[210]
	print(obs)
	print(f[19])
