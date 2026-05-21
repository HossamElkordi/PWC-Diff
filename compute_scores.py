import os
import cv2
import json
import torch
import pyiqa
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str)                 
    parser.set_defaults(ref=True)                      
    args = parser.parse_args()
    return args


def score_full_ref(metrics, metric_list, dname, preds_dir, gts_dir):
    def compute_full(metrics, metric_list, pred, pred_clipped, gt):
        scores = dict()
        for met in metric_list:
            score = metrics[met](pred, gt).item()
            score_adj = metrics[met](pred_clipped, gt).item()
            scores[met] = score
            scores[f'{met}_adj'] = score_adj
        return scores
    
    final_scores = []
    gts_fnames = sorted(os.listdir(gts_dir))
    preds_fnames = sorted(os.listdir(preds_dir))
    for g, p in tqdm(zip(gts_fnames, preds_fnames), desc=f'Computing Full Reference Scores For {dname}', total=len(gts_fnames), colour='green'):
        gt_fpath = os.path.join(gts_dir, g)
        pred_fpath = os.path.join(preds_dir, p)
        
        rimg = Image.open(gt_fpath)
        fimg = Image.open(pred_fpath)
        
        mean_gray_out = cv2.cvtColor(np.array(fimg).astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_gray_gt = cv2.cvtColor(np.array(rimg).astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        
        rimg = pil_to_tensor(rimg).to(torch.float).unsqueeze(0).transpose(3, 2) / 255
        fimg = pil_to_tensor(fimg).to(torch.float).unsqueeze(0).transpose(3, 2) / 255
        fimg_clipped = torch.clip(fimg * (mean_gray_gt / mean_gray_out), 0, 1)
        rimg = rimg[:, :, :fimg.shape[2], :fimg.shape[3]]
        sample_scores = dict(id=g)
        sample_scores.update(compute_full(metrics, metric_list, fimg, fimg_clipped, rimg))
        final_scores.append(sample_scores)
    return pd.DataFrame(final_scores)


def score_no_ref(metrics, metric_list, dname, preds_dir):
    def compute_no(metrics, metric_list, pred):
        scores = dict()
        for met in metric_list:
            score = metrics[met](pred).item()
            scores[met] = score
        return scores
    
    final_scores = []
    preds_fnames = sorted(os.listdir(preds_dir))
    for p in tqdm(preds_fnames, desc=f'Computing No Reference Scores For {dname}', total=len(preds_fnames), colour='green'):
        pred_fpath = os.path.join(preds_dir, p)
        sample_scores = dict(id=p)
        sample_scores.update(compute_no(metrics, metric_list, pred_fpath))
        final_scores.append(sample_scores)
    return pd.DataFrame(final_scores)
        
        

if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as fp:
        config = json.load(fp)
        
    save_detailed_scores = config['save_detailed_scores']
    save_dir = config['save_dir']
    metrics = {
        met: pyiqa.create_metric(met, device='cpu', as_loss=False)
        for met in config['full_metrics'] + config['no_metrics']
    }
    
    no_reference_config = config['no_ref']
    full_reference_config = config['full_ref']
    
    agg_full_scores = []
    for data in full_reference_config:
        dname = data['name']
        gts_dir = data['gts_dir']
        preds_dir = data['preds_dir']
        
        scores = score_full_ref(metrics, config['full_metrics'], dname, preds_dir, gts_dir)
        if save_detailed_scores:
            scores.to_csv(os.path.join(save_dir, f'{dname}_full_scores.csv'), index=False)
        
        all_metrics = config['full_metrics'] + [f'{met}_adj' for met in config['full_metrics']]
        data_scores = dict(dname=dname)
        data_scores.update(scores[all_metrics].mean().to_dict())
        agg_full_scores.append(data_scores)
        
    if len(agg_full_scores) > 0:
        print(agg_full_scores)
        
    agg_no_scores = []
    for data in no_reference_config:
        dname = data['name']
        preds_dir = data['preds_dir']
        
        scores = score_no_ref(metrics, config['no_metrics'], dname, preds_dir)
        if save_detailed_scores:
            scores.to_csv(os.path.join(save_dir, f'{dname}_no_scores.csv'), index=False)
        
        data_scores = dict(dname=dname)
        data_scores.update(scores[config['no_metrics']].mean().to_dict())
        agg_no_scores.append(data_scores)
    
    if len(agg_no_scores) > 0:
        print(agg_no_scores)