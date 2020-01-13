from CNN_model import CNN_Model
import torch
from dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import os

dataloader = DataLoader(Dataset('train-data'),
	batch_size = 32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
f_dim = 3
model = CNN_Model(f_dim).to(device)

model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(10):

	model.train()
	bar = tqdm(dataloader)

	total_loss, total = 0, 0

	for Il, Ir, label in bar:

		optimizer.zero_grad()

		# plt.subplot(1, 2, 1)
		# plt.title(str(label[0]))
		# plt.imshow(Il[0])
		# plt.subplot(1, 2, 2)
		# plt.imshow(Ir[0])
		# plt.show()
		
		Il = Il.to(device)
		Ir = Ir.to(device)
		label = label.to(device)

		# channel first
		Il = Il.permute(0, 3, 1, 2)
		Ir = Ir.permute(0, 3, 1, 2)

		# print(Il.shape, Il.dtype)
		Fl = model(Il)
		Fr = model(Ir)

		h, w = Fl.shape[-2:]
		Fl = Fl[:, :, h//2, w//2]
		Fr = Fr[:, :, h//2, w//2]

		Fl_len = torch.norm(Fl, dim=1).detach()
		Fl /= Fl_len.unsqueeze(1)
		Fr_len = torch.norm(Fr, dim=1).detach()
		Fr /= Fr_len.unsqueeze(1)

		score = (Fl.unsqueeze(1) @ Fr.unsqueeze(2)).squeeze()

		ops_score = score[label == 1].mean()
		neg_score = score[label == 0].mean()

		m = 0.2

		loss = m+neg_score-ops_score
		total_loss += loss.item()
		total += 1

		loss.backward()
		optimizer.step()

		bar.set_postfix(
			ops = '{:.02f}'.format(ops_score.item()),
			neg = '{:.02f}'.format(neg_score.item()),
			loss = '{:.04f}'.format(total_loss/total)
		)
	
	torch.save(model.state_dict(), os.path.join(model_dir, f'{epoch}.pth'))

	