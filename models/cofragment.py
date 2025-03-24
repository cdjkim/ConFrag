import os
from copy import deepcopy
import tqdm
import torch
import pickle as pkl
import torch.nn.functional as F
import colorful
import numpy as np
import random
from collections import defaultdict
from tensorboardX import SummaryWriter

from components import Net
from utils import CRITERION, write_grad_norm_scalar


class CoFragment(torch.nn.Module):
    """ Coteach + Fragment
    """
    def __init__(self, config, scheduler, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.writer = writer
        self.scheduler = scheduler

        assert len(self.config['label_split']) == 1
        self.experts_1, self.experts_subset_info = self.get_init_experts(config)
        self.experts_2, _ = self.get_init_experts(config)
        self.criterion = CRITERION[config['loss']](config, writer)

        self.filter_save_path = os.path.join(config['log_dir'], 'filter')
        os.makedirs(self.filter_save_path, exist_ok=True)

        self.cval_history = []
        self.test_history = []

        ### Co-teach hyperparam
        # exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.
        self.exponent = self.config['exponent']
        self.noise_rate = self.config['noise']['corrupt_p'] # corruption rate should be less than
        forget_rate = self.noise_rate
        # n_gradual: how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.
        n_gradual = self.config['n_gradual']

        # define drop rate schedule
        self.rate_schedule = np.ones(self.config['task_epochs']) * forget_rate
        self.rate_schedule[:n_gradual] = np.linspace(0, forget_rate ** self.exponent, n_gradual)
        ###

    def get_init_experts(self, config):
        """get initialized expert models"""
        experts = []
        schedule = self.scheduler.schedule
        experts_subset_info = []
        for i in range(len(schedule)):
            if config['loss'] == 'mse':
                expert = Net[config['net']](config)
            else:
                nb_classes = len(schedule[i]['subsets'])
                expert = Net["resnet_classify"](config, nb_classes=nb_classes)
            optim_config = config['optimizer']
            lr_scheduler_config = deepcopy(config['lr_scheduler'])
            if lr_scheduler_config['type'] == 'CosineAnnealingLR':
                # lr_scheduler_config['options'].update({'T_max': schedule[i]['epoch']})
                lr_scheduler_config['options'].update({'T_max': config['task_epochs']})

            expert.setup_optimizer(optim_config)
            expert.setup_lr_scheduler(lr_scheduler_config)

            expert.train()
            experts.append(expert)
            experts_subset_info.append(schedule[i]['subsets'])

        print(colorful.bold_green(f'num of experts {len(experts)}').styled_string)

        return experts, experts_subset_info

    def get_eval_model(self, t):
        expert = self.experts_1[t]
        expert.eval()
        return expert

    def get_eval_dataset(self, t):
        test_dataset = self.scheduler.test_task_datasets[t]
        val_dataset = self.scheduler.val_task_datasets[t]
        clf_y_dim = self.scheduler.task_i_to_dim[t]
        return test_dataset, val_dataset, clf_y_dim

    def learn(self, task, epoch, t, step):
        expert_1 = self.experts_1[t]
        expert_2 = self.experts_2[t]
        task_real_labels = self.scheduler.schedule[t]['subsets']

        for epc_i in tqdm.trange(epoch, desc='epoch', leave=False):
            self.epoch_label_precision = []
            self.pred_clean_idcs_save = []
            correct_1, correct_2 = 0, 0
            total = 0
            for batch in tqdm.tqdm(task, desc='step', leave=False):
                if self.config['loss'] == 'ce':
                    x, clf_y, clf_gt_y, y, gt_y, idx = batch
                    x, clf_y, clf_gt_y = x.to(self.device), clf_y[0].to(self.device), clf_gt_y[0].to(self.device)
                else:
                    x, y, gt_y, idx = batch
                    clf_y, clf_gt_y = None, None
                    x, y, gt_y = x.to(self.device), y.to(self.device), gt_y.to(self.device)

                expert_1.zero_grad(set_to_none=True)
                expert_2.zero_grad(set_to_none=True)

                y = torch.unsqueeze(y, dim=1)
                y1_hat = expert_1(x)
                y2_hat = expert_2(x)

                loss_1, loss_2 = self.loss_coteach(y1_hat, y2_hat, clf_y, clf_gt_y, y, gt_y, step, idx, self.rate_schedule[epc_i], t)

                expert_1.optimizer.zero_grad(set_to_none=True)
                loss_1.backward()
                expert_1.optimizer.step()

                expert_2.optimizer.zero_grad(set_to_none=True)
                loss_2.backward()
                expert_2.optimizer.step()

                self.writer.add_scalar(f'train/loss_{t}', (loss_1+loss_2)/2, step)

                step += 1

            self.writer.add_scalar(f'train/Label_Precision_{t}/epc', sum(self.epoch_label_precision)/len(self.epoch_label_precision), epc_i)
            if (epc_i+1)%self.config['eval_every'] == 0 or epc_i == epoch:
                print(colorful.bold_white(f'eval epoch {epc_i}, step {step}').styled_string)
                self.scheduler.eval(self, self.writer, epc_i, t, task_real_labels)

            if (epc_i+1)%self.config['save_every'] == 0 or epc_i+1 == epoch:
                name_format = str(epc_i + 1)
                clean_idcs_save_path = os.path.join(self.filter_save_path, f'clean_idcs_{name_format}_{t}.pkl')
                with open(clean_idcs_save_path, 'wb') as handle:
                    pkl.dump(self.pred_clean_idcs_save, handle, protocol=pkl.HIGHEST_PROTOCOL)

        # based on val's best epoch, return test's corresponding epoch's value.
        best_val = min(self.cval_history)
        best_val_epoch = self.cval_history.index(best_val)
        best_test_from_val = self.test_history[best_val_epoch]
        self.writer.add_scalar('final/val/val_mae', best_val, 0)
        self.writer.add_scalar('final/val/test_mae', best_test_from_val, 0)
        print(colorful.bold_green(f'final/val/val_mae: {best_val}'))
        print(colorful.bold_green(f'final/val/test_mae: {best_test_from_val}'))

        return step


    def forward(self, x):
        """ Used at eval time.
        """
        if self.config['loss'] == 'ce':
            return self.base1(x)
        else:
            y1_hat = self.base1(x)
            y2_hat = self.base2(x)
            return torch.mean(torch.cat((y1_hat, y2_hat), dim=1), dim=1).unsqueeze(dim=1)

    def loss_coteach(self, y1_hat, y2_hat, clf_y, clf_gt_y, y, gt_y, step, idx, forget_rate, t):

        with torch.no_grad():
            if self.config['loss'] == 'ce':
                loss_pick_1 = F.cross_entropy(y1_hat, clf_y, reduction = 'none')
                loss_pick_2 = F.cross_entropy(y2_hat, clf_y, reduction = 'none')
            else:
                loss_pick_1 = F.mse_loss(y1_hat, y, reduction = 'none')
                loss_pick_2 = F.mse_loss(y2_hat, y, reduction = 'none')

            loss_pick_1 = loss_pick_1.squeeze()
            loss_pick_2 = loss_pick_2.squeeze()
            #ind1_sorted = np.argsort(loss_pick_1.data)
            ind1_sorted = torch.argsort(loss_pick_1)
            ind2_sorted = torch.argsort(loss_pick_2)

            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * self.config['batch_size'])

            # select top num_remember small losses.
            ind1_update = ind1_sorted[:num_remember].tolist()
            ind2_update = ind2_sorted[:num_remember]

            self.pred_clean_idcs_save.extend(idx[ind1_update])
            # filter based on loss ranking
            # the number of clean samples in here would tell us, performance of filtering.
            # optimal scenario: noisy samples incurs a higher loss, and hence be ranked lower.
            # filter based on loss ranking
            if self.config['classification']:
                clf_gt_y = clf_gt_y.squeeze()
                clf_y = clf_y.squeeze()
                gt_y_update1 = clf_gt_y[ind1_update]
                y_update1 = clf_y[ind1_update]

                gt_y_update2 = clf_gt_y[ind2_update]
                y_update2 = clf_y[ind2_update]

                noise_mask = clf_y == clf_gt_y
            else:
                gt_y_update1 = gt_y[ind1_update]
                y = y.squeeze()
                y_update1 = y[ind1_update]

                gt_y_update2 = gt_y[ind2_update]
                y_update2 = y[ind2_update]
                noise_mask = y == gt_y

            n_correct1 = torch.sum(gt_y_update1 == y_update1)
            n_correct2 = torch.sum(gt_y_update2 == y_update2)
            if self.config['noise']['corrupt_p'] > 0.0:
                # print wrong label loss and clean label loss separately.
                clean_idcs = torch.nonzero(noise_mask, as_tuple=True)[0]
                noisy_idcs = torch.nonzero(torch.logical_not(noise_mask), as_tuple=True)[0]
                clean_loss1 = torch.mean(loss_pick_1[clean_idcs])
                noisy_loss1 = torch.mean(loss_pick_1[noisy_idcs])
                clean_loss2 = torch.mean(loss_pick_2[clean_idcs])
                noisy_loss2 = torch.mean(loss_pick_2[noisy_idcs])
                self.writer.add_scalar(f'train/clean_loss_{t}', (clean_loss1 + clean_loss2)/2, step)
                self.writer.add_scalar(f'train/noisy_loss_{t}', (noisy_loss1 + noisy_loss2)/2, step)

        y1_hat = y1_hat.squeeze()
        y2_hat = y2_hat.squeeze()
        if self.config['loss'] == 'ce':
            loss_1_update = F.cross_entropy(y1_hat[ind2_update], clf_y[ind2_update], reduction = 'mean')
            loss_2_update = F.cross_entropy(y2_hat[ind1_update], clf_y[ind1_update], reduction = 'mean')
        else:
            loss_1_update = F.mse_loss(y1_hat[ind2_update], y[ind2_update], reduction = 'mean')
            loss_2_update = F.mse_loss(y2_hat[ind1_update], y[ind1_update], reduction = 'mean')

        # Label_Precision = (# of clean labels) / (# of all selected labels).
        label_precision = (n_correct1 + n_correct2)/(num_remember*2)
        self.writer.add_scalar(f'train/Label_Precision_{t}/step', label_precision, step)
        self.epoch_label_precision.append(label_precision)

        return loss_1_update, loss_2_update
