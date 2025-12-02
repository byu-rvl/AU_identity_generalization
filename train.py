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

from pathlib import Path
import tarfile
import shutil

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def untar_fsrt_data(tar_path, extract_path):
    Path(extract_path).mkdir(parents=True, exist_ok=True)
    # untar the tar file to /tmp/{jobID}/fsrt_data/epoch_{epoch}/
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extracted FSRT data to:", extract_path)

def get_train_dataloader(conf, epoch):
    epoch = int(epoch) + 1 # for zero indexing.

    un_tar_path = f"/tmp/{conf.jobID}/prep_fsrt/epoch{epoch}"

    # Delete previous epoch's extracted data to save space
    delete_tar_path = f"/tmp/{conf.jobID}/prep_fsrt/epoch{epoch-1}"
    if os.path.exists(delete_tar_path):
        shutil.rmtree(delete_tar_path)
        print("Deleted previous epoch's extracted FSRT data at:", delete_tar_path)

    print('==> Preparing data...')
    from_bp4d = f"{conf.fsrt_dataset_path}/BP4D_epoch{epoch}.tar.gz"
    to_bp4d = f"{un_tar_path}/BP4D_epoch{epoch}"
    from_disfa = f"{conf.fsrt_dataset_path}/DISFA_epoch{epoch}.tar.gz"
    to_disfa = f"{un_tar_path}/DISFA_epoch{epoch}"
    if conf.dataset == 'BP4D':
        untar_fsrt_data(from_bp4d, to_bp4d)
        trainset = BP4D(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf, epoch=epoch, fsrt_dataset_path=to_bp4d)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, worker_init_fn=seed_worker)

    elif conf.dataset == 'DISFA':
        untar_fsrt_data(from_disfa, to_disfa)
        trainset = DISFA(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf, epoch=epoch, fsrt_dataset_path=to_disfa)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, worker_init_fn=seed_worker)

    elif conf.dataset == 'FEC':
        trainset = FEC(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 2, conf=conf)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, drop_last=True, worker_init_fn=seed_worker)

    elif conf.dataset == 'both':
        untar_fsrt_data(from_bp4d, to_bp4d)
        untar_fsrt_data(from_disfa, to_disfa)
        bp4d_trainset = BothDatasets(do_dataset="bp4d", root_path_bp4d=conf.bp4d_dataset_path, root_path_disfa=conf.disfa_dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf, epoch=epoch, fsrt_dataset_path=to_bp4d)
        bp4d_trainloader = DataLoader(bp4d_trainset, batch_size=conf.batch_size//2, shuffle=True, num_workers=conf.num_workers, worker_init_fn=seed_worker)
        disfa_trainset = BothDatasets(do_dataset="disfa", root_path_bp4d=conf.bp4d_dataset_path, root_path_disfa=conf.disfa_dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1, conf=conf, epoch=epoch, fsrt_dataset_path=to_disfa)
        disfa_trainloader = DataLoader(disfa_trainset, batch_size=conf.batch_size//2, shuffle=True, num_workers=conf.num_workers, worker_init_fn=seed_worker)
        train_loader = CombinedDataLoader(bp4d_trainloader, disfa_trainloader)

    return train_loader

def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, worker_init_fn=seed_worker)

    elif conf.dataset == 'DISFA':
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, worker_init_fn=seed_worker)

    elif conf.dataset == 'FEC':
        valset = FEC(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2, conf=conf)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, worker_init_fn=seed_worker)

    elif conf.dataset == 'both':
        bp4d_valset = BothDatasets(do_dataset="bp4d", root_path_bp4d=conf.bp4d_dataset_path, root_path_disfa=conf.disfa_dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
        disfa_valset = BothDatasets(do_dataset="disfa", root_path_bp4d=conf.bp4d_dataset_path, root_path_disfa=conf.disfa_dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1, conf=conf)
        bp4d_valloader = DataLoader(bp4d_valset, batch_size=conf.batch_size//2, shuffle=False, num_workers=conf.num_workers, worker_init_fn=seed_worker)
        disfa_valloader = DataLoader(disfa_valset, batch_size=conf.batch_size//2, shuffle=False, num_workers=conf.num_workers, worker_init_fn=seed_worker)
        val_loader = CombinedDataLoader(bp4d_valloader, disfa_valloader)
        valset = bp4d_valset + disfa_valset

    return val_loader, len(valset)

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
def train(conf,net,optimizer,epoch,criterion):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    net.train()

    train_loader = get_train_dataloader(conf, epoch)
    train_loader_len = len(train_loader)

    translate_every_other = False

    for batch_idx, data in enumerate(tqdm(train_loader)):
        if conf.dataset == "both":
            data, dataset_flags = data    
        
        inputs, aug_inputs, targets, relations, lmk_true = data
        fsrt_batch_size = inputs.shape[0]
        # concatente inputs and aug_inputs in the batch dimension
        inputs = torch.cat((inputs, aug_inputs), 0)
        # targets = torch.cat((targets, targets), 0)
        # relations = torch.cat((relations, relations), 0)
        # lmk_true = torch.cat((lmk_true, lmk_true), 0)

        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        targets = targets.float()
        lmk_true = lmk_true.float()
        if torch.cuda.is_available():
            inputs, targets, relations, lmk_true = inputs.cuda(), targets.cuda(), relations.cuda(), lmk_true.cuda()
        optimizer.zero_grad()
        outputs, outputs_relation, emb_out, lmk_out = net(inputs)
        
        if conf.dataset == "both" and dataset_flags == "bp4d":
            normal_outputs = outputs[:fsrt_batch_size]
            fsrt_outputs = outputs[fsrt_batch_size:]
            loss_normal = criterion[0]["bp4d"](normal_outputs, targets, is_fsrt=False)
            loss_fsrt = criterion[0]["bp4d"](fsrt_outputs, targets, is_fsrt=True)
            loss = loss_normal + loss_fsrt
        elif conf.dataset == "both" and dataset_flags == "disfa":
            normal_outputs = outputs[:fsrt_batch_size]
            fsrt_outputs = outputs[fsrt_batch_size:]
            loss_normal = criterion[0]["disfa"](normal_outputs, targets, is_fsrt=False)
            loss_fsrt = criterion[0]["disfa"](fsrt_outputs, targets, is_fsrt=True)
            loss = loss_normal + loss_fsrt
        else:
            normal_outputs = outputs[:fsrt_batch_size]
            fsrt_outputs = outputs[fsrt_batch_size:]
            loss_normal = criterion[0](normal_outputs, targets, is_fsrt=False)
            loss_fsrt = criterion[0](fsrt_outputs, targets, is_fsrt=True)
            loss = loss_normal + loss_fsrt
        
        output_relations_normal = outputs_relation[:fsrt_batch_size]
        output_relations_fsrt = outputs_relation[fsrt_batch_size:]

        # edge_loss_normal = criterion[1](output_relations_normal.view(-1,4), relations.view(-1).long())
        # edge_loss_fsrt = criterion[1](output_relations_fsrt.view(-1,4), relations.view(-1).long())
        # edge_loss = edge_loss_normal + edge_loss_fsrt
        contrasitive_loss_normal = criterion[2](emb_out[:fsrt_batch_size], targets)
        contrasitive_loss_fsrt = criterion[2](emb_out[fsrt_batch_size:], targets)
        contrasitive_loss = contrasitive_loss_normal + contrasitive_loss_fsrt

        loss = loss + conf.lam_contrasitive * contrasitive_loss


        # edge_loss = criterion[1](outputs_relation.view(-1,4), relations.view(-1).long())
        # edge_loss = criterion[1](outputs_relation.view(-1,4), relations.view(-1).long())
        # contrasitive_loss = criterion[2](emb_out, targets)
        # lmk_loss = criterion[3](lmk_out.float(), lmk_true)
        # loss = wa_loss + conf.lam_edge * edge_loss + conf.lam_contrasitive * contrasitive_loss + conf.lam_lmk * lmk_loss
        
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
        # if conf.dataset != "both":
        #     losses1.update(wa_loss.data.item(), inputs.size(0))
        #     losses2.update(edge_loss.data.item(), inputs.size(0))
        #     losses3.update(contrasitive_loss.data.item(), inputs.size(0))
        # else:
        losses1.update(0.0, inputs.size(0))
        losses2.update(0.0, inputs.size(0))
        losses3.update(0.0, inputs.size(0))
    return losses.avg, losses1.avg, losses2.avg, losses3.avg


# Val
def val(net,val_loader,criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            if conf.dataset == "both":
                data, dataset_flags = data
            inputs, targets = data
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _, _ = net(inputs)
            if conf.dataset == "both" and dataset_flags == "bp4d":
                loss = criterion[0]["bp4d"](outputs, targets)
            elif conf.dataset == "both" and dataset_flags == "disfa":
                loss = criterion[0]["disfa"](outputs, targets)
            else:
                loss = criterion[0](outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics(outputs, targets.detach(), 0.5)
            if conf.dataset == "both":
                statistics_list = update_statistics_list(statistics_list, update_list, dataset_flag=dataset_flags)
            else:
                statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


def main(conf):
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For full determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    elif conf.dataset == 'both':
        dataset_info = None
        numberLmks=66 #TODO: if I use landmarks, I will need to update this.

    start_epoch = 0
    # data
    val_loader,val_data_num = get_dataloader(conf)
    if conf.dataset != "FEC":
        if conf.dataset == 'both':
            bp4d_class_totals = np.loadtxt('/fslhome/andreww9/code/original_datasets/BP4D_croppped_MTCNN/list/BP4D_class_totals_fold'+str(conf.fold)+'.txt')
            disfa_class_totals = np.loadtxt('/home/andreww9/fsl_groups/grp_RVL_AU/code/DISFA_LEFT_cropped_MTCNN/list/DISFA_class_totals_fold'+str(conf.fold)+'.txt')

            # BP4D has 12 AUs: 1,2,4,6,7,10,12,14,15,17,23,24
            # DISFA has 8 AUs: 1,2,4,6,9,12,25,26
            # Together they have 15 unique AUs: 1,2,4,6,7,9,10,12,14,15,17,23,24,25,26
            all_class_totals = [0] * 15
            # Map BP4D class totals
            BP4D_indices = [0,1,2,3,4,6,7,8,9,10,11,12]
            DISFA_indices = [0,1,2,3,5,7,13,14]
            for i in range(len(all_class_totals)):
                if i in BP4D_indices:
                    bp4d_index = BP4D_indices.index(i)
                    all_class_totals[i] += bp4d_class_totals[bp4d_index]
                if i in DISFA_indices:
                    disfa_index = DISFA_indices.index(i)
                    all_class_totals[i] += disfa_class_totals[disfa_index]

            all_class_totals = np.array(all_class_totals)
            all_class_totals = all_class_totals / all_class_totals.sum()
            all_class_weights = 1.0 / all_class_totals
            
            bp4d_weight = []
            disfa_weight = []
            for i in range(len(all_class_weights)):
                if i in BP4D_indices:
                    bp4d_weight.append(all_class_weights[i])
                if i in DISFA_indices:
                    disfa_weight.append(all_class_weights[i])
            bp4d_weight = np.array(bp4d_weight)
            disfa_weight = np.array(disfa_weight)
            bp4d_weight = bp4d_weight / bp4d_weight.sum() * bp4d_weight.shape[0]
            disfa_weight = disfa_weight / disfa_weight.sum() * disfa_weight.shape
            bp4d_weight = torch.from_numpy(bp4d_weight)
            disfa_weight = torch.from_numpy(disfa_weight)

            # raise Exception("Need to implement class weights for both datasets.")
        else:
            train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', conf.dataset+'_weight_fold'+str(conf.fold)+'.txt')))

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc, numEncoderLayers=conf.numEncoderLayers, numLandmarks=numberLmks, conf=conf)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        if conf.dataset == 'both':
            bp4d_weight = bp4d_weight.cuda()
            disfa_weight = disfa_weight.cuda()
        elif conf.dataset != "FEC":
            train_weight = train_weight.cuda()

    margin = 0.2
    if conf.dataset == "FEC":
        criterion = [TripleContrasitiveLoss(margin=margin)]
    elif conf.dataset == 'both':
        main_criterion = {"bp4d": WeightedAsymmetricLoss(weight=bp4d_weight, dataset="bp4d", smoothing=conf.smoothing, limitFsrtLoss=conf.limitFsrtLoss), "disfa": WeightedAsymmetricLoss(weight=disfa_weight, dataset="disfa", smoothing=conf.smoothing, limitFsrtLoss=conf.limitFsrtLoss)}
        criterion = [main_criterion, nn.CrossEntropyLoss(),BatchContrastiveLoss(margin=margin),nn.MSELoss()]
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
            train_loss, wa_loss, edge_loss, lmk_loss = train(conf,net,optimizer,epoch,criterion)
            val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)

            # log
            infostr = {'Epoch:  {}   train_loss: {:.5f} wa_loss: {:.5f} edge_loss: {:.5f} lmk_loss: {:.5f} val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                    .format(epoch + 1, train_loss, wa_loss, edge_loss, lmk_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}

            logging.info(infostr)
            infostr = {'F1-score-list:'}
            logging.info(infostr)
            if dataset_info is not None:
                infostr = dataset_info(val_f1_score)
            else:
                infostr = f"val_f1_score: {val_f1_score}"
            logging.info(infostr)
            infostr = {'Acc-list:'}
            logging.info(infostr)
            if dataset_info is not None:
                infostr = dataset_info(val_acc)
            else:
                infostr = f"val_acc: {val_acc}"
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