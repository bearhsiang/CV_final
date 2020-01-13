from util import readPFM
import os
import sys
import cv2
import numpy as np

def resize(img, w):
	H, W = img.shape[:2]
	h = int(H * w/W)
	return cv2.resize(img, (h, w))

def main():

	data_dir = sys.argv[1]
	output_dir = 'train-data'
	os.makedirs(output_dir, exist_ok=True)

	with open(os.path.join(output_dir, 'train.csv'), 'w') as f:
		print('lb,rb,label', file=f)
		total = 0
		# Il = cv2.imread(os.path.join(data_dir, 'im0.png'))
		# Ir = cv2.imread(os.path.join(data_dir, 'im1.png'))
		for i in range(10):
			print(i)
			Il = cv2.imread(os.path.join(data_dir, f'TL{i}.png'))
			Ir = cv2.imread(os.path.join(data_dir, f'TR{i}.png'))
		
			# l_d_map = readPFM(os.path.join(data_dir, 'disp0.pfm'))
			# r_d_map = readPFM(os.path.join(data_dir, 'disp1.pfm'))

			l_d_map = readPFM(os.path.join(data_dir, f'TLD{i}.pfm'))
			
			H, W, C = Il.shape

			r = 10
			N = 1000
			count = 0
			d_ops_list = [-1, 0, 1]
			d_neg_list = []
			d_neg_list.extend(np.arange(-21, -10))
			d_neg_list.extend(np.arange(11, 22))
			while True:

				h = np.random.randint(r, H-r)
				w = np.random.randint(r, W-r)

				if l_d_map[h, w] == np.inf:
					continue

				d = int(l_d_map[h, w])
				w_r = w - d

				# if r_d_map[h, w_r] == np.inf:
				# 	continue

				# if abs(r_d_map[h, w_r] - d) > 2:
				# 	continue

				ops_w_r = w_r + d_ops_list[np.random.randint(len(d_ops_list))]
				neg_w_r = w_r + d_neg_list[np.random.randint(len(d_neg_list))]

				if ops_w_r < r or ops_w_r > W-r-1 or neg_w_r < r or neg_w_r > W-r-1:
					continue

				l_patch = Il[h-r:h+r+1, w-r:w+r+1]
				ops_r_patch = Ir[h-r:h+r+1, ops_w_r-r:ops_w_r+r+1]
				neg_r_patch = Ir[h-r:h+r+1, neg_w_r-r:neg_w_r+r+1]

				l_name = f'{total}_l.png'
				ops_r_name = f'{total}_ops_r.png'
				neg_r_name = f'{total}_neg_r.png'
				cv2.imwrite(os.path.join(output_dir, l_name), l_patch)
				cv2.imwrite(os.path.join(output_dir, ops_r_name), ops_r_patch)
				cv2.imwrite(os.path.join(output_dir, neg_r_name), neg_r_patch)

				print(f'{l_name},{ops_r_name},1', file=f)
				print(f'{l_name},{neg_r_name},0', file=f)

				count += 1
				total += 1
				if count >= N:
					break


if __name__ == '__main__':
	main()