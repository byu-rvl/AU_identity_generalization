import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import glob
import imageio
import yaml
from easydict import EasyDict as edict

from model.encoder_gcn import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env

from third_party_helpers.access_fsrt import access_fsrt

fsrt_model = access_fsrt()

def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.eval_dataset == 'BP4D':
        with open('config/BP4D_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
        valset = BP4D(datasets_cfg.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.eval_dataset == 'DISFA':
        with open('config/DISFA_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
        print(datasets_cfg)
        print(datasets_cfg.dataset_path)
        valset = DISFA(datasets_cfg.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return val_loader, len(valset)

# Val
def val(net,val_loader):
    net.eval()
    statistics_list = None
    statistics_list_overlap = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _, _ = net(inputs)
            
            # BP4D AUs are 1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24
            # DISFA AUs are 1, 2, 4, 6, 9, 12, 25, 26
            # Overlapping AUs are 1, 2, 4, 6, 12
            disfa_overlap_indices = [0,1,2,3,5]  # AUs 1,2,4,6,12
            bp4d_overlap_indices = [0,1,2,3,7]

            if conf.dataset == conf.eval_dataset:
                update_list = statistics(outputs, targets.detach(), 0.5)
                statistics_list = update_statistics_list(statistics_list, update_list)
                
                if conf.dataset == 'BP4D':
                    update_list_overlap = statistics(outputs[:, bp4d_overlap_indices], targets[:, bp4d_overlap_indices].detach(), 0.5)
                    statistics_list_overlap = update_statistics_list(statistics_list_overlap, update_list_overlap)
                elif conf.dataset == 'DISFA':
                    update_list_overlap = statistics(outputs[:, disfa_overlap_indices], targets[:, disfa_overlap_indices].detach(), 0.5)
                    statistics_list_overlap = update_statistics_list(statistics_list_overlap, update_list_overlap)

            if conf.dataset == 'BP4D' and conf.eval_dataset == 'DISFA':
                outputs_overlap = outputs[:, bp4d_overlap_indices]
                targets_overlap = targets[:, disfa_overlap_indices]
                update_list = statistics(outputs_overlap, targets_overlap.detach(), 0.5)
                statistics_list_overlap = update_statistics_list(statistics_list_overlap, update_list)
            elif conf.dataset == 'DISFA' and conf.eval_dataset == 'BP4D':
                outputs_overlap = outputs[:, disfa_overlap_indices]
                targets_overlap = targets[:, bp4d_overlap_indices]
                update_list = statistics(outputs_overlap, targets_overlap.detach(), 0.5)
                statistics_list_overlap = update_statistics_list(statistics_list_overlap, update_list)
            
    if statistics_list is not None:
        mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
        mean_acc, acc_list = calc_acc(statistics_list)
    else:
        mean_f1_score, f1_score_list, mean_acc, acc_list = None, None, None, None

    mean_f1_score_overlap, f1_score_list_overlap = calc_f1_score(statistics_list_overlap)
    mean_acc_overlap, acc_list_overlap = calc_acc(statistics_list_overlap)

    return mean_f1_score, f1_score_list, mean_acc, acc_list, mean_f1_score_overlap, f1_score_list_overlap, mean_acc_overlap, acc_list_overlap


def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
        numberLmks=49
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist
        numberLmks=66

    # data
    val_loader,val_data_num = get_dataloader(conf)

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc, numEncoderLayers=conf.numEncoderLayers, numLandmarks=numberLmks)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)
    else:
        raise Exception("No resume file provided.")

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    margin = 0.2

    #train and val
    logging.info("Starting Validation")
    val_mean_f1_score, val_f1_score, val_mean_acc, val_acc, val_mean_f1_score_overlap, val_f1_score_overlap, val_mean_acc_overlap, val_acc_overlap = val(net, val_loader)

    if conf.dataset == conf.eval_dataset:
        # log
        infostr = {'Validation: val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                .format(100.* val_mean_f1_score, 100.* val_mean_acc)}

        logging.info(infostr)
        infostr = {'F1-score-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_f1_score)
        logging.info(infostr)
        infostr = {'Acc-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_acc)
        logging.info(infostr)
    
    infostr = {'Validation on overlapping AUs: val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
            .format(100.* val_mean_f1_score_overlap, 100.* val_mean_acc_overlap)}
    logging.info(infostr)
    infostr = {'F1-score-list:'}
    logging.info(infostr)
    infostr = overlap_infolist(val_f1_score_overlap)
    logging.info(infostr)
    infostr = {'Acc-list:'}
    logging.info(infostr)
    infostr = overlap_infolist(val_acc_overlap)
    logging.info(infostr)


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)