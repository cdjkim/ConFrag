import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import joblib
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import math
from abc import abstractmethod

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from tensorboardX import SummaryWriter


Tensor = torch.Tensor

class Criterion(object):
    def __init__(self, config:dict, writer: SummaryWriter):
        self.config = config
        self.criterion = config['loss']
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.writer = writer

    @abstractmethod
    def __call__(self, input: Tensor, target: Tensor, gt_target=None, clean_mask=None,
                 reduction: str='mean', idx=None, test=False, epoch=None, step=None) -> Tensor:
        pass

    def flush(self, epoch, postfix=''):
        return None

class SimpleLoss(Criterion):
    """
    loss init and inference
    """
    def __init__(self, config: dict, writer: SummaryWriter):
        super(SimpleLoss, self).__init__(config, writer)
        if self.criterion == 'bmse':
            self.init_bmse()

        elif self.criterion == 'weighted_mse':
            self.normalize_batch = self.config['loss_params']['weighted_mse']['normalize_batch']

    def __call__(self, input: Tensor, target: Tensor, gt_target=None, clean_mask=None,
                 reduction: str='mean', idx=None, test=False, epoch=None, step=None) -> Tensor:
        if self.criterion in ['mse', 'l2']:
            target = target.view(-1, 1)
            return F.mse_loss(input, target, reduction=reduction)

        elif self.criterion in ['l1']:
            target = target.view(-1, 1)
            return F.l1_loss(input, target, reduction=reduction)

        elif self.criterion in ['ce', 'crossentropy']:
            target = target.to(torch.int64)
            return F.cross_entropy(input, target, reduction=reduction)

        elif self.criterion in ['sce']:
            return self.sce_loss(input, target, reduction, test=test)

        else:
            raise NotImplementedError(f"NotImplemented loss type {self.criterion}")


    def sce_loss(self, input: Tensor, target: Tensor, reduction: str, test=False):
        """
        (Wang, ICCV 2019) Symmetric Cross Entropy for Robust Learning with Noisy Labels
        https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
        l_sl = alpha*l_ce + beta*l_rce
        A = log(eps_rce) -> -4 following paper
        """
        self.alpha = self.config['loss_params']['sce']['alpha']
        self.beta = self.config['loss_params']['sce']['beta']
        y_hat_1 = F.softmax(input, dim=1)
        y_1 =  F.one_hot(target, num_classes=input.shape[1])

        y_hat_2 = y_hat_1
        y_2 = y_1

#        y_hat_1 = torch.clamp(y_hat_1, min=1e-7, max=1.0)
        y_2 = torch.clamp(y_2, min=1e-4, max=1.0)

        loss = -self.alpha * torch.sum(y_1 * torch.log(y_hat_1), dim=1) \
            -self.beta * torch.sum(y_2 * torch.log(y_hat_2), dim=1)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError(f"NotImplemented reduction type: {reduction}")


class SceLoss(Criterion):
    """
    (Wang, ICCV 2019) Symmetric Cross Entropy for Robust Learning with Noisy Labels
    https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
    l_sl = alpha*l_ce + beta*l_rce
    A = log(eps_rce) -> -4 following paper
    """
    def __call__(self, input: Tensor, target: Tensor, gt_target=None, clean_mask=None,
                 reduction: str='mean', idx=None, test=False, epoch=None, step=None) -> Tensor:
        self.alpha = self.config['loss_params']['sce']['alpha']
        self.beta = self.config['loss_params']['sce']['beta']
        y_hat = F.softmax(input, dim=1)
        y =  F.one_hot(target, num_classes=input.shape[1])

        y_hat_clamped = torch.clamp(y_hat, min=1e-7, max=1.0)
        y_clamped = torch.clamp(y, min=1e-4, max=1.0) # min=1e-4 from A=-4

        loss_ce = torch.sum(y * torch.log(y_hat_clamped), dim=1)
        loss_rce = torch.sum(y_hat * torch.log(y_clamped), dim=1)
        loss = - self.alpha * loss_ce - self.beta * loss_rce

        if step != None:
            self.writer.add_scalar('sce/train/loss_ce', torch.mean(loss_ce), step)
            self.writer.add_scalar('sce/train/loss_rce', torch.mean(loss_rce), step)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError(f"NotImplemented reduction type: {reduction}")


class SiguaLoss(Criterion):
    """
    (Han, ICML 2022) SIGUA: Forgetting May Make Learning with Noisy Labels More Robust
    https://github.com/bhanML/SIGUA
    * only SIGUA_SL is implemented since SIGUA_BC requires the noise transition matrix
    """
    def __init__(self, config: dict, writer: SummaryWriter, dataset=None):
        super(SiguaLoss, self).__init__(config, writer)
        assert config['loss'] == 'sigua_mse'
        self.loss_fn = F.mse_loss
        self.sum_loss = True

        self.noise_p = self.config['noise']['corrupt_p']
        self.warmup_epoch = self.config['loss_params']['sigua']['warmup_epoch']
        self.sigua_rate = self.config['loss_params']['sigua']['sigua_rate']
        self.sigua_scale = self.config['loss_params']['sigua']['sigua_scale']

        self.num_select = defaultdict(int)
        self.num_sigua = defaultdict(int)
        self.num_total = defaultdict(int)

        self.num_clean = defaultdict(int)
        self.num_tp = defaultdict(int)
        self.num_correct = defaultdict(int)
        self.error_select = defaultdict(float)

        self.dataset = dataset
        self.soft_num_clean = defaultdict(int)
        self.soft_num_tp = defaultdict(int)
        self.soft_num_correct = defaultdict(int)
        self.soft_error_select = defaultdict(float)

        self.pred_clean_idc = defaultdict(list)

    def __call__(self, input: Tensor, target: Tensor, gt_target=None, clean_mask=None,
                 reduction: str='mean', test=False, idx=None, epoch=None, step=None) -> Tensor:
        if epoch == None or self.warmup_epoch == 0:
            forget_rate = self.noise_p
        else:
            forget_rate = self.noise_p * min(epoch/self.warmup_epoch, 1)

        loss = self.loss_fn(input, target, reduction='none')
        if self.sum_loss:
            loss = torch.sum(loss, dim=1, keepdim=True)
        ind_sorted = torch.argsort(torch.squeeze(loss)) # from small to big

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss))
        num_forget = len(loss) - num_remember
        num_sigua = int(num_forget * self.sigua_rate)

        small_thld = loss[ind_sorted[num_remember - 1]]
        loss_small = torch.le(loss, small_thld) * loss

        if num_sigua > 0:
            big_thld = loss[ind_sorted[num_remember + num_sigua - 1]]
        else:
            big_thld = loss[ind_sorted[num_remember - 1]]
        loss_big = torch.gt(loss, small_thld) * torch.le(loss, big_thld) * loss

        loss_aggr = loss_small - self.sigua_scale * loss_big

        if epoch!=None:
            clean_mask = clean_mask.to(self.device)
            self.num_sigua[epoch] += num_sigua
            self.num_total[epoch] += len(target)
            self.num_clean[epoch] += int(torch.sum(clean_mask).item())

            small_mask = torch.le(loss, small_thld).squeeze()
            self.num_select[epoch] += torch.sum(small_mask) # compute small mask this way for the case of equal loss
            self.num_tp[epoch] += int(torch.sum(small_mask * clean_mask).item())
            self.num_correct[epoch] += int(torch.sum(small_mask == clean_mask).item())

            self.error_select[epoch] += torch.sum(torch.abs(gt_target - target)[small_mask])

            #for gaussian_random
            soft_clean_mask = (torch.abs(gt_target - target) < (self.dataset.range_target/8)).squeeze()
            self.soft_num_clean[epoch] += int(torch.sum(soft_clean_mask))
            self.soft_num_tp[epoch] += int(torch.sum(small_mask * soft_clean_mask).item())

            error_select = torch.abs(gt_target - target)[small_mask]
            soft_error_select = error_select[(error_select > (self.dataset.range_target/8))]

            self.soft_num_correct[epoch] += torch.sum(small_mask == soft_clean_mask)
            self.soft_error_select[epoch] += torch.sum(soft_error_select)

            if idx != None:
                self.pred_clean_idc[epoch].extend(idx[small_mask.cpu()].tolist())

        if reduction == 'mean':
            return torch.mean(loss_aggr)
        elif reduction == 'none':
            return loss_aggr
        else:
            raise NotImplementedError(f"NotImplemented reduction type: {reduction}")

    def flush(self, epoch, postfix=''):
        ###################################################################
        # print shared tensorbard
        #   - filter/filter_percentage
        #   - filter/filter_error
        #   - filter/refurb_percentage
        #   - filter/refurb_error
        #   - filter/select_percentage
        #   - filter/select_error
        ###################################################################

        selection_ratio = self.num_select[epoch] / self.num_total[epoch]
        precision = self.num_tp[epoch] / self.num_select[epoch] if self.num_select[epoch] != 0 else 0
        recall = self.num_tp[epoch] / self.num_clean[epoch] if self.num_clean[epoch] != 0 else 0
        f1_score = 2*precision*recall/(precision + recall) if self.num_clean[epoch] != 0 else 0
        accuracy = self.num_correct[epoch] / self.num_total[epoch]

        filter_error = self.error_select[epoch] / self.num_select[epoch] if self.num_select[epoch] != 0 else 0

        self.writer.add_scalar(f'filter/filter_percentage{postfix}', selection_ratio, epoch)
        self.writer.add_scalar(f'filter/filter_error{postfix}', filter_error, epoch)
        self.writer.add_scalar(f'filter/select_percentage{postfix}', selection_ratio, epoch)
        self.writer.add_scalar(f'filter/select_error{postfix}', filter_error, epoch)
        self.writer.add_scalar(f'filter/clean_precision{postfix}', precision, epoch)
        self.writer.add_scalar(f'filter/clean_recall{postfix}', recall, epoch)

        #######################################################################
        # for gaussian_noise
        soft_precision = self.soft_num_tp[epoch] / self.num_select[epoch] if self.num_select[epoch] != 0 else 0
        soft_recall = self.soft_num_tp[epoch] / self.soft_num_clean[epoch] if self.soft_num_clean[epoch] != 0 else 0
        soft_f1_score = 2*soft_precision*soft_recall/(soft_precision + soft_recall) if self.soft_num_clean[epoch] != 0 else 0
        soft_accuracy = self.soft_num_correct[epoch] / self.num_total[epoch]

        soft_filter_error = self.soft_error_select[epoch] / self.num_select[epoch] if self.num_select[epoch] != 0 else 0

        self.writer.add_scalar(f'filter/soft_filter_percentage{postfix}', selection_ratio, epoch)
        self.writer.add_scalar(f'filter/soft_filter_error{postfix}', soft_filter_error, epoch)
        self.writer.add_scalar(f'filter/soft_select_percentage{postfix}', selection_ratio, epoch)
        self.writer.add_scalar(f'filter/soft_select_error{postfix}', soft_filter_error, epoch)
        self.writer.add_scalar(f'filter/soft_clean_precision{postfix}', soft_precision, epoch)
        self.writer.add_scalar(f'filter/soft_clean_recall{postfix}', soft_recall, epoch)

        curr_pred_clean_idc = self.pred_clean_idc[epoch]
        # reset pred_clean_idc to save memory
        self.pred_clean_idc[epoch] = []
        return curr_pred_clean_idc


CRITERION = {
    'mse': SimpleLoss,
    'l2': SimpleLoss,
    'l1': SimpleLoss,
    'ce': SimpleLoss,
    'sce': SceLoss,
    'sigua_mse': SiguaLoss,
}
