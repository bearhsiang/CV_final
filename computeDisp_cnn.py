import numpy as np
import cv2
import torch
import torch.nn.functional as F
from baseline import create_features, estimate_cost, estimate_disparity, holefilling
from CNN_model import CNN_Model

def preprocess(Il, Ir):
    
    energy_Il = np.sum(Il)
    energy_Ir = np.sum(Ir)

    Ir = Ir.astype(np.float32) * (energy_Il/energy_Ir)
    Il = Il.astype(np.float32)

    return Il.astype(np.uint8), Ir.astype(np.uint8)

def img2tensor(img):
    img = img.astype(np.float32)/255
    img = torch.tensor(img).permute(2, 0, 1)
    return img

def computeDisp(Il, Ir, max_disp=50, do_refine=True, model_path='model/9.pth'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN_Model(in_dim = 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    Il, Ir = preprocess(Il, Ir)
    
    Il_tensor = img2tensor(Il).to(device).unsqueeze(0)
    Ir_tensor = img2tensor(Ir).to(device).unsqueeze(0)

    features_l = model(Il_tensor).squeeze().permute(1, 2, 0)
    features_r = model(Ir_tensor).squeeze().permute(1, 2, 0)

    costs = []

    for d in range(max_disp):

        ### shift features
        if d > 0:
            shifted_features_r = F.pad(features_r[:, :-d, :], (0, 0, d, 0), 'constant', 0)
        else:
            shifted_features_r = features_r

        ### estimate cost
        # cost = estimate_cost(features_l, shifted_features_r)
        h, w, c = features_r.shape
        cost = torch.bmm(features_l.view(-1, 1, c), shifted_features_r.view(-1, c, 1))
        cost = cost.view(h,w)
        cost = cost.detach().numpy()
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
