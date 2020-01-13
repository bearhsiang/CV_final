from torch.utils import data
import cv2
import os
import torch
from baseline import create_features
import numpy as np

class Dataset(data.Dataset):
	def __init__(self, data_dir):
		super().__init__()
		self.data_list = []
		with open(os.path.join(data_dir, 'train.csv'), 'r') as f:
			first = True
			for s in f:
				if first:
					first = False
					continue
				Il, Ir, label = s.strip().split(',')
				Il = os.path.join(data_dir, Il)
				Ir = os.path.join(data_dir, Ir)
				self.data_list.append([Il, Ir, label])


	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		Il = cv2.imread(self.data_list[idx][0]).astype(np.float32)/255
		# Il = create_features(Il)/255
		Ir = cv2.imread(self.data_list[idx][1]).astype(np.float32)/255
		# Ir = create_features(Ir)/255
		label = int(self.data_list[idx][2])
		return Il, Ir, label