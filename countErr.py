from util import cal_avgerr, readPFM
import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--GT', default='./data/Synthetic/TLD0.pfm', type=str, help='Ground Truth')
    parser.add_argument('--input', default='./TL0.pfm', type=str, help='input disparity map')
    args = parser.parse_args()

    GT = readPFM(args.GT)
    disp = readPFM(args.input)

    print(cal_avgerr(GT, disp))

