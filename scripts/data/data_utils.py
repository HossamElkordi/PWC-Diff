import os
import glob
import torch
import numpy as np
from scipy import fft
from skimage import exposure

def get_illumination_map(map_name, x, dtype):
    if map_name == 'ours':
        return ours_map(x)
    elif map_name == 'lime':
        return lime_illumination_map(x, dtype=dtype)
    else:
        raise NotImplementedError

def ours_map(x):
    return 1.0 - x.max(dim=0, keepdim=True).values

def lime_illumination_map(L, dtype=torch.float32, iterations=10, alpha=2, rho=2, gamma=0.7, strategy=2):
    def firstOrderDerivative(n, k=1):
        return np.eye(n) * (-1) + np.eye(n, k=k)

    def toeplitizMatrix(n, row):
        vecDD = np.zeros(n)
        vecDD[0] = 4
        vecDD[1] = -1
        vecDD[row] = -1
        vecDD[-1] = -1
        vecDD[-row] = -1
        return vecDD

    def vectorize(matrix):
        return matrix.T.ravel()

    def reshape(vector, row, col):
        return vector.reshape((row, col), order='F')

    if torch.is_tensor(L):
        L = L.detach().cpu().permute(1, 2, 0).numpy()

    row, col, _ = L.shape
    T_hat = np.max(L, axis=2)

    dv = firstOrderDerivative(row)
    dh = firstOrderDerivative(col, -1)
    vecDD = toeplitizMatrix(row * col, row)

    if strategy == 2:
        dTv = dv @ T_hat
        dTh = T_hat @ dh
        Wv = 1 / (np.abs(dTv) + 1)
        Wh = 1 / (np.abs(dTh) + 1)
        W = np.vstack([Wv, Wh])
    else:
        W = np.ones((row * 2, col))

    T = np.zeros((row, col))
    G = np.zeros((row * 2, col))
    Z = np.zeros((row * 2, col))
    u = 1

    for _ in range(iterations):
        X = G - Z / u
        Xv = X[:row, :]
        Xh = X[row:, :]
        temp = dv @ Xv + Xh @ dh
        numerator = fft.fft(vectorize(2 * T_hat + u * temp))
        denominator = fft.fft(vecDD * u) + 2
        T = fft.ifft(numerator / denominator)
        T = np.real(reshape(T, row, col))
        T = exposure.rescale_intensity(T, (0, 1), (0.001, 1))

        v = dv @ T
        h = T @ dh
        dT = np.vstack([v, h])
        epsilon = alpha * W / u
        X = dT + Z / u
        G = np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

        Z = Z + u * (dT - G)
        u *= rho

    return torch.from_numpy(T ** gamma).to(dtype).unsqueeze(0)

def exists(x):
    return x is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))

def get_sid_data(config):
    short_dir = config['short_dir']
    long_dir = config['long_dir']
    phase = config['phase']
    
    subfolders_LQ_origin = glob_file_list(short_dir)
    subfolders_GT_origin = glob_file_list(long_dir)
    
    subfolders_LQ = []
    subfolders_GT = []
    if phase == 'train':
        for mm in range(len(subfolders_LQ_origin)):
            name = os.path.basename(subfolders_LQ_origin[mm])
            if '0' in name[0] or '2' in name[0]:
                subfolders_LQ.append(subfolders_LQ_origin[mm])
                subfolders_GT.append(subfolders_GT_origin[mm])
    else:
        for mm in range(len(subfolders_LQ_origin)):
            name = os.path.basename(subfolders_LQ_origin[mm])
            if '1' in name[0]:
                subfolders_LQ.append(subfolders_LQ_origin[mm])
                subfolders_GT.append(subfolders_GT_origin[mm])
    
    data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
    imgs_LQ, imgs_GT = {}, {} 
    for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
        # for frames in each video:
        subfolder_name = os.path.basename(subfolder_LQ)

        img_paths_LQ = glob_file_list(subfolder_LQ)
        img_paths_GT = glob_file_list(subfolder_GT)

        max_idx = len(img_paths_LQ)
        data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
        data_info['path_GT'].extend(img_paths_GT)
        data_info['folder'].extend([subfolder_name] * max_idx)
        for i in range(max_idx):
            data_info['idx'].append('{}/{}'.format(i, max_idx))

        border_l = [0] * max_idx
        for i in range(5 // 2):
            border_l[i] = 1
            border_l[max_idx - i - 1] = 1
        data_info['border'].extend(border_l)


        imgs_LQ[subfolder_name] = img_paths_LQ
        imgs_GT[subfolder_name] = img_paths_GT
        
    x_paths, y_paths = [], []
    for i in range(len(data_info['path_LQ'])):
        folder = data_info['folder'][i]
        idx, max_idx = data_info['idx'][i].split('/')
        idx, max_idx = int(idx), int(max_idx)

        img_LQ_path = imgs_LQ[folder][idx]
        img_LQ_path = [img_LQ_path]
        img_GT_path = imgs_GT[folder][0]
        img_GT_path = [img_GT_path]
        
        x_paths.extend(img_LQ_path)
        y_paths.extend(img_GT_path)
    
    return x_paths, y_paths
