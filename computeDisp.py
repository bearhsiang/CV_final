import numpy as np
import cv2
from cv2.ximgproc import *


from baseline import create_features, estimate_cost, estimate_disparity, holefilling

def preprocess(Il, Ir):
    
    energy_Il = np.sum(Il)
    energy_Ir = np.sum(Ir)

    return Il, (Ir.astype(float) * energy_Il/energy_Ir).astype(np.uint8)

def computeDisp(Il, Ir, max_disp = 100, do_refine=True):
    
    Il, Ir = preprocess(Il, Ir)

    features_l = np.array(create_features(Il))
    features_r = np.array(create_features(Ir))

    costs = []

    for d in range(max_disp):
        
        ### shift features
        if d > 0:
            shifted_features_r = np.pad(features_r[:, :, :-d], ((0, 0), (0, 0), (d, 0)), mode='edge')
        else:
            shifted_features_r = features_r

        ### estimate cost
        cost = estimate_cost(features_l, shifted_features_r)
        
        ### weighted filter
        cost = cv2.ximgproc.guidedFilter(Il, cost, 11, 7)
        
        costs.append(cost)

    costs = np.array(costs)

    d_map = estimate_disparity(costs)

    if do_refine:
        r_d_map = computeDisp(np.flip(Ir, 1), np.flip(Il, 1), max_disp=max_disp, do_refine=False)
        r_d_map = np.flip(r_d_map, 1)

        # Left-Right check
        d_map = holefilling(d_map, r_d_map).astype(np.float32)
        ## weighted median filter
        
        d_map = cv2.ximgproc.weightedMedianFilter(Il, d_map.astype(np.uint8), 5)
        # d_map = cv2.ximgproc.guidedFilter(Il, d_map, 11, 7)


    return d_map.astype(np.float32)
