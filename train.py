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

from model.encoder_gcn import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env

def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'FEC':
        trainset = FEC(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 2, conf=conf)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, drop_last=True)
        valset = FEC(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2, conf=conf)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    # elif conf.dataset == 'Both':
    #     with open('config/BP4D_config.yaml', 'r') as f:
    #         datasets_cfg = yaml.safe_load(f)
    #         datasets_cfg = edict(datasets_cfg)
    #     trainset_BP4D = BP4D(datasets_cfg.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf)
    #     valset_BP4D = BP4D(datasets_cfg.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
    #     val_loader_BP4D = DataLoader(valset_BP4D, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    #     with open('config/DISFA_config.yaml', 'r') as f:
    #         datasets_cfg = yaml.safe_load(f)
    #         datasets_cfg = edict(datasets_cfg)
    #     trainset_DISFA = DISFA(datasets_cfg.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf)
    #     valset_DISFA = DISFA(datasets_cfg.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
    #     val_loader_DISFA = DataLoader(valset_DISFA, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    #     trainset = torch.utils.data.ConcatDataset([trainset_BP4D, trainset_DISFA])
    #     train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    #     val_loader = [val_loader_BP4D, val_loader_DISFA]

    return train_loader, val_loader, len(trainset), len(valset)

def train_FEC(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (imgs0, imgs1, imgs2, label) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        if torch.cuda.is_available():
            imgs0, imgs1, imgs2, label = imgs0.cuda(), imgs1.cuda(), imgs2.cuda(), label.cuda()
        optimizer.zero_grad()
        _, _, emb_out0, _ = net(imgs0)
        _, _, emb_out1, _ = net(imgs1)
        _, _, emb_out2, _ = net(imgs2)
        emb_out0 = emb_out0.unsqueeze(1)
        emb_out1 = emb_out1.unsqueeze(1)
        emb_out2 = emb_out2.unsqueeze(1)
        outputs = torch.cat((emb_out0, emb_out1, emb_out2), 1)
        # print("shapes: ", outputs.shape, label.shape, emb_out0.shape, emb_out1.shape, emb_out2.shape)
        loss = criterion[0](outputs, label)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), outputs.size(0))

    return losses.avg


def val_FEC(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    losses = AverageMeter()
    for batch_idx, (imgs0, imgs1, imgs2, label) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            if torch.cuda.is_available():
                imgs0, imgs1, imgs2, label = imgs0.cuda(), imgs1.cuda(), imgs2.cuda(), label.cuda()

            _, _, emb_out0, _ = net(imgs0)
            _, _, emb_out1, _ = net(imgs1)
            _, _, emb_out2, _ = net(imgs2)
            emb_out0 = emb_out0.unsqueeze(1)
            emb_out1 = emb_out1.unsqueeze(1)
            emb_out2 = emb_out2.unsqueeze(1)
            outputs = torch.cat((emb_out0, emb_out1, emb_out2), 1)
            loss = criterion[0](outputs, label)
            statistics_list = update_statistics_list_FEC(statistics_list, outputs, label)
            losses.update(loss.data.item(), outputs.size(0))
    num_correct = statistics_list[0]
    num_incorrect = statistics_list[1]
    total = statistics_list[2]
    mean_acc = num_correct / total
    return losses.avg, mean_acc

# Train
def train(conf,net,train_loader,optimizer,epoch,criterion):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)

    translate_every_other = False

    for batch_idx, (inputs,  targets, relations, lmk_true) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        targets = targets.float()
        lmk_true = lmk_true.float()
        if torch.cuda.is_available():
            inputs, targets, relations, lmk_true = inputs.cuda(), targets.cuda(), relations.cuda(), lmk_true.cuda()
        optimizer.zero_grad()
        outputs, outputs_relation, emb_out, lmk_out = net(inputs)
        wa_loss = criterion[0](outputs, targets)
        edge_loss = criterion[1](outputs_relation.view(-1,4), relations.view(-1).long())
        contrasitive_loss = criterion[2](emb_out, targets)
        lmk_loss = criterion[3](lmk_out.float(), lmk_true)
        loss = wa_loss + conf.lam_edge * edge_loss + conf.lam_contrasitive * contrasitive_loss + conf.lam_lmk * lmk_loss
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
        losses1.update(wa_loss.data.item(), inputs.size(0))
        losses2.update(edge_loss.data.item(), inputs.size(0))
        losses3.update(contrasitive_loss.data.item(), inputs.size(0))
    return losses.avg, losses1.avg, losses2.avg, losses3.avg


# Val
def val(net,val_loader,criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _, _ = net(inputs)
            loss = criterion[0](outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
        numberLmks=49
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist
        numberLmks=66
    elif conf.dataset == "FEC":
        dataset_info = FEC_infolist
        # numberLmks=49
        numberLmks=66

    start_epoch = 0
    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)
    if conf.dataset != "FEC":
        train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', conf.dataset+'_weight_fold'+str(conf.fold)+'.txt')))

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc, numEncoderLayers=conf.numEncoderLayers, numLandmarks=numberLmks, conf=conf)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        if conf.dataset != "FEC":
            train_weight = train_weight.cuda()

    margin = 0.2
    if conf.dataset == "FEC":
        criterion = [TripleContrasitiveLoss(margin=margin)]
    else:
        criterion = [WeightedAsymmetricLoss(weight=train_weight), nn.CrossEntropyLoss(),BatchContrastiveLoss(margin=margin),nn.MSELoss()]
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        if conf.dataset == "FEC":
            train_loss = train_FEC(conf,net,train_loader,optimizer,epoch,criterion)
            val_loss, val_mean_acc = val_FEC(net, val_loader, criterion)
            infostr = {'Epoch:  {}   train_loss: {:.5f} val_loss: {:.5f}  val_mean_acc {:.2f}'
                    .format(epoch + 1, train_loss, val_loss, 100.* val_mean_acc)}
            logging.info(infostr)
        else:
            train_loss, wa_loss, edge_loss, lmk_loss = train(conf,net,train_loader,optimizer,epoch,criterion)
            val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)

            # log
            infostr = {'Epoch:  {}   train_loss: {:.5f} wa_loss: {:.5f} edge_loss: {:.5f} lmk_loss: {:.5f} val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                    .format(epoch + 1, train_loss, wa_loss, edge_loss, lmk_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}

            logging.info(infostr)
            infostr = {'F1-score-list:'}
            logging.info(infostr)
            infostr = dataset_info(val_f1_score)
            logging.info(infostr)
            infostr = {'Acc-list:'}
            logging.info(infostr)
            infostr = dataset_info(val_acc)
            logging.info(infostr)

        # save checkpoints
        if (epoch+1) % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold) + '.pth'))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) # ADD THESE TWO LINES
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)