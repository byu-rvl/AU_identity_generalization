from math import cos, pi
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def statistics(pred, y, thresh):
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if y[i][j] == 9:  # ignore label, skip. This is the flag from the EBplus dataset when face is out of the frame.
                continue

            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list

def update_statistics_list_FEC(statistics_list, outputs, label):
    diff_first_second = torch.norm(outputs[:,0,:] - outputs[:,1,:], p=2, dim=1)
    diff_first_third = torch.norm(outputs[:,0,:] - outputs[:,2,:], p=2, dim=1)
    diff_second_third = torch.norm(outputs[:,1,:] - outputs[:,2,:], p=2, dim=1)

    combined = torch.cat((torch.unsqueeze(diff_second_third,1), torch.unsqueeze(diff_first_third,1), torch.unsqueeze(diff_second_third,1)), 1)
    predictions = torch.argmin(combined, 1)

    # add 1 to all prediction values so indexing is the same as the labels
    predictions += 1

    correct = predictions == label
    incorrect = ~correct
    num_correct = torch.sum(correct, dim=0)
    num_incorrect = torch.sum(incorrect, dim=0)
    total = num_correct + num_incorrect

    
    if statistics_list is None:
        statistics_list = [num_correct, num_incorrect, total]
    else:
        statistics_list = [statistics_list[0] + num_correct, statistics_list[1] + num_incorrect, statistics_list[2] + total]
    return statistics_list

def BP4D_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}
    return infostr

def DISFA_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    return infostr

def EBplus_infolist(list):
    # list has AUs: 1,2,4,6,7,10,12,14,15,17,23,24, so 12 AUs
    infostr = 'EBplus: AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])
    # BP4D AUs are 1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24
    # Overlapping AUs are 1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24 (all BP4D are in EBplus)
    infostr += "\nBP4D Overlap is the same as EBplus line above. "
    # DISFA AUs are 1, 2, 4, 6, 9, 12, 25, 26
    # Overlapping AUs are 1, 2, 4, 6, 12
    disfa_indices_in_ebplus = [0,1,2,3,6]
    infostr += '\nDISFA Overlap (same as 5 normal overlap): AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU12: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[6])
    infostr += '\n(Note: EBplus does not have DISFA AUs 9, 25, 26). '
    infostr += '\nThe cross 5 AU overlap average (AUs 1,2,4,6,12) F1 score between EBplus and DISFA is: {:.2f}'.format(100.*(sum([list[i] for i in disfa_indices_in_ebplus])/len(disfa_indices_in_ebplus)))
    return {infostr}

def overlap_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU12: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4])}
    return infostr

def FEC_infolist(list):
    infostr = "insert info str in utils.py"
    return infostr


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class image_train(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


class image_test(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model


class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None, dataset=None, smoothing=0.0):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight
        self.dataset = dataset

        self.bp4d_indicies = [0,1,2,3,4,6,7,8,9,10,11,12]
        self.disfa_indicies = [0,1,2,3,5,7,13,14]

        self.smoothing = smoothing

    def forward(self, x, y):

        if self.dataset == "bp4d":
            x = x[:, self.bp4d_indicies]
            y = y[:, self.bp4d_indicies]
        elif self.dataset == "disfa":
            x = x[:, self.disfa_indicies]
            y = y[:, self.disfa_indicies]

        if self.smoothing > 0:
            x[x<self.smoothing] = 0.0
            x[x>1-self.smoothing] = 1.0

        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()

class BatchContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, inv_prop_pad=5):
        super(BatchContrastiveLoss, self).__init__()
        self.margin = margin
        self.prop_pad_add = 1 - 1/inv_prop_pad
        numAU = 8
        self.divideAmnt = numAU * inv_prop_pad

    def forward(self, outputs, labels):
        # Calculate pairwise distances for all pairs in the batch
        pairwise_distances = torch.cdist(outputs, outputs, p=2)

        # calculate L1 distance labels from all other labels
        pairwise_distances_labels = torch.cdist(labels, labels, p=1) #p is for L1 distance

        # make it go between 0 and 1, but weighted so that it will be be 0 for the same item, and between self.prop_pad_add and 1 for the different items.
        # pairwise_distances_labels = pairwise_distances_labels / self.divideAmnt + self.prop_pad_add
        # pairwise_distances_labels[pairwise_distances_labels==self.prop_pad_add] = 0.0
        pairwise_distances_labels[pairwise_distances_labels!=0]=1.0

        # Create a mask for positive and negative pairs
        positive_mask = 1 - pairwise_distances_labels
        negative_mask = pairwise_distances_labels

        # Calculate the contrastive loss
        loss_contrastive = torch.sum(positive_mask * torch.pow(pairwise_distances, 2) +
                                     negative_mask * torch.pow(torch.clamp(self.margin - pairwise_distances, min=0.0),
                                                               2)) / 2.0
        return loss_contrastive

class TripleContrasitiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripleContrasitiveLoss, self).__init__()
        self.margin = 0.1#margin

    def forward(self, outputs, labels):

        diff_first_second = torch.norm(outputs[:,0,:] - outputs[:,1,:], p=2, dim=1)
        diff_first_third = torch.norm(outputs[:,0,:] - outputs[:,2,:], p=2, dim=1)
        diff_second_third = torch.norm(outputs[:,1,:] - outputs[:,2,:], p=2, dim=1)
        labels_is1 = labels == 1
        labels_is2 = labels == 2
        labels_is3 = labels == 3

        # margin = 0.2
        # loss_1 = (labels_is1) * (diff_second_third**2) + (~labels_is1) * (torch.max(torch.tensor(0.0), margin - diff_second_third)**2)
        # loss_2 = (labels_is2) * (diff_first_third**2) + (~labels_is2) * (torch.max(torch.tensor(0.0), margin - diff_first_third)**2)
        # loss_3 = (labels_is3) * (diff_first_second**2) + (~labels_is3) * (torch.max(torch.tensor(0.0), margin - diff_first_second)**2)

        margin = 0.1
        loss_1 = (labels_is1) * (torch.max(torch.tensor(0.0), diff_second_third - diff_first_second + margin)**2 + torch.max(torch.tensor(0.0), diff_second_third - diff_first_third + margin)**2)
        loss_2 = (labels_is2) * (torch.max(torch.tensor(0.0), diff_first_third - diff_first_second + margin)**2 + torch.max(torch.tensor(0.0), diff_first_third - diff_second_third + margin)**2)
        loss_3 = (labels_is3) * (torch.max(torch.tensor(0.0), diff_first_second - diff_first_third + margin)**2 + torch.max(torch.tensor(0.0), diff_first_second - diff_second_third + margin)**2)

        all_loss = loss_1 + loss_2 + loss_3
        return torch.sum(all_loss, dim=0)

class CombinedDataLoader:
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2

    def __iter__(self):
        self.iter1 = iter(self.dataloader1)
        self.iter2 = iter(self.dataloader2)
        self.iter1_exhausted = False
        self.iter2_exhausted = False
        self.use_bp4d_next = True
        return self

    def __next__(self):
        if self.iter1_exhausted and self.iter2_exhausted:
            raise StopIteration

        # Determine which loader to use, preferring alternation
        # but falling back to the non-exhausted one.
        use_bp4d = (self.use_bp4d_next and not self.iter1_exhausted) or self.iter2_exhausted
        
        try:
            if use_bp4d and not self.iter1_exhausted:
                data = next(self.iter1)
                dataset_flag = "bp4d"
            elif not self.iter2_exhausted:
                data = next(self.iter2)
                dataset_flag = "disfa"
            else:
                # This case should not be hit due to the initial check, but as a safeguard:
                raise StopIteration

            # Alternate for the next call if both are still active
            if not self.iter1_exhausted and not self.iter2_exhausted:
                self.use_bp4d_next = not self.use_bp4d_next
            
            return data, dataset_flag

        except StopIteration as e:
            if use_bp4d:
                self.iter1_exhausted = True
            else:
                self.iter2_exhausted = True
            
            # Retry getting the next batch from the other loader in the same call
            return self.__next__()

    def __len__(self):
        return len(self.dataloader1) + len(self.dataloader2)
