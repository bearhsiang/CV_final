import torch
import torch.nn as nn

class CNN_Model(nn.Module):

	def __init__(self, in_dim):
		super().__init__()
		self.cnn = nn.Sequential(

			nn.Conv2d(in_dim, 64, 3, padding=1),
			nn.ReLU(),

			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(),


			# nn.Conv2d(32, 32, 3, padding=1),
			# nn.ReLU(),

			nn.Conv2d(64, 64, 3, padding=1),
		)
	
	def forward(self, img):
		fmap = self.cnn(img)
		return fmap
