import numpy as np
import cv2

def create_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = np.expand_dims(cv2.Sobel(gray, cv2.CV_32F, 1, 0), axis=0)
    grad_y = np.expand_dims(cv2.Sobel(gray, cv2.CV_32F, 0, 1), axis=0)
    
    features = np.concatenate((np.moveaxis(img, -1, 0), grad_x, grad_y), axis=0)

    return features

def estimate_cost(features_l, features_r):
    
    diff = np.abs(features_l - features_r)
    diff_I = np.sum(diff[:3], axis=0)
    diff_g = np.sum(diff[3:], axis=0)

    alpha = 0.5

    return (1-alpha)*diff_I + alpha * diff_g

def estimate_disparity(costs):
    return np.argmin(costs, axis=0).astype(np.float32)

def holefilling(d_map, r_d_map):

    h, w = d_map.shape

    ### hole filling
    d_map_hat = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            d_map_hat[i][j] = r_d_map[i][int(j-d_map[i][j])]
    valid_mask = (d_map - d_map_hat) == 0

    N = np.amax(d_map)+1

    left_scan = np.full((h, w), N)
    record = np.full(h, N)
    for i in range(w):
        record[valid_mask[:, i]] = d_map[valid_mask[:, i], i]
        left_scan[:, i] = record
    

    right_scan = np.full((h, w), N)
    record = np.full(h, N)
    for i in range(w-1, -1, -1):
        record[valid_mask[:, i]] = d_map[valid_mask[:, i], i]
        right_scan[:, i] = record
    
    d_map = np.minimum(left_scan, right_scan)

    return d_map
