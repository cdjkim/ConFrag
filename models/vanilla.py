import os, sys
from copy import deepcopy
import tqdm
import torch
import colorful
import pickle as pkl
import numpy as np
from tensorboardX import SummaryWriter
from components import Net
from utils import CRITERION, write_grad_norm_scalar
sys.path.append('../')
from utils import get_exp_path, count_parameters


def update_taskloader(dataset, epc_i, config, writer):
    if config['filter_load'] == '':
        saved_filter_path = os.path.join(config['log_dir'], 'filter')
    else:
        saved_filter_path = get_exp_path(config['filter_load'], 'filter', config['use_recent_identifier'])

    if 'cofragment' in config['log_dir'] or 'xiafragment' in config['log_dir']:
        clean_idcs = []
        for t, _ in enumerate(config['data_schedule']):
            saved_clean_idcs_path = os.path.join(saved_filter_path, f"clean_idcs_{epc_i}_{t}.pkl")
            with open(saved_clean_idcs_path, 'rb') as handle:
                clean_idcs.extend(pkl.load(handle))
            print(f'load filter idcs from {saved_filter_path}')
    else:
        saved_clean_idcs_path = os.path.join(saved_filter_path, f"clean_idcs_{epc_i}.pkl")
        with open(saved_clean_idcs_path, 'rb') as handle:
            clean_idcs = pkl.load(handle)
        print(f'load filter idcs from {saved_filter_path}')

    # reset dataset targets
    dataset.reset_targets()
    len_total_idcs = len(dataset.targets)

    if len(clean_idcs) == 0:
        print(f'No clean_idcs at current epoch {epc_i}')
        return None

    # only using clean_idcs
    dataset.update_data(clean_idcs)
    writer.add_histogram('train/filtered_label', dataset.targets[clean_idcs], epc_i)
    writer.add_histogram('train/updated_label', dataset.targets[clean_idcs], epc_i)

    print(colorful.bold_green(f'filter_percentage {len(clean_idcs)/len_total_idcs:.4f}'
                              ).styled_string)

    ###################################################################
    # print shared tensorbard
    #   - filter/filter_percentage
    #   - filter/filter_error
    ###################################################################

    filter_percentage = len(clean_idcs)/len_total_idcs
    filter_error = np.sum(np.abs(dataset.targets - dataset.org_targets)[clean_idcs])

    filter_error = filter_error / len(clean_idcs) if len(clean_idcs) != 0 else 0

    writer.add_scalar(f'filter/filter_percentage', filter_percentage, epc_i)
    writer.add_scalar(f'filter/filter_error', filter_error, epc_i)

    soft_filter_percentage = len(clean_idcs)/len_total_idcs

    soft_error = np.abs((dataset.targets - dataset.org_targets) > (dataset.range_target/8))
    soft_filter_error = np.sum(soft_error[clean_idcs])

    soft_filter_error = soft_filter_error / len(clean_idcs) if len(clean_idcs) != 0 else 0

    writer.add_scalar(f'filter/soft_filter_percentage', soft_filter_percentage, epc_i)
    writer.add_scalar(f'filter/soft_filter_error', soft_filter_error, epc_i)
    ###################################################################

    writer.add_scalar('filter/filter_percentage', len(clean_idcs)/len_total_idcs, epc_i)
#    writer.add_scalar('filter/dataset_error', np.mean(np.abs(dataset.targets - dataset.org_targets)), epc_i)
    writer.add_scalar('filter/dataset_error_on_updated_idcs',
                      np.mean(np.abs(dataset.targets[dataset.total_idcs] - dataset.org_targets[dataset.total_idcs])), epc_i)

    loader = torch.utils.data.DataLoader(
        dataset.subsets['all'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True, # better when training on GPU.
        shuffle=True)

    return loader


class Vanilla(torch.nn.Module):
    def __init__(self, config, scheduler, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.writer = writer
        self.scheduler = scheduler
        self.dataset = scheduler.datasets[self.config['data_name']]

        self.criterion = CRITERION[config['loss']](config, writer)
        if 'sigua' in self.config['loss']:
            self.criterion = CRITERION[config['loss']](config, writer, self.dataset)

        self.base = self.get_init_base(config)

        # for general setting
        self.nval_history = []
        self.cval_history = []
        self.test_history = []

    def get_init_base(self, config):
        """get initialized base model"""

        base = Net[config['net']](config)
        optim_config = config['optimizer']
        lr_scheduler_config = deepcopy(config['lr_scheduler'])
        if lr_scheduler_config['type'] == 'CosineAnnealingLR':
            lr_scheduler_config['options'].update({'T_max': config['total_epochs']})

        base.setup_optimizer(optim_config)
        base.setup_lr_scheduler(lr_scheduler_config)

        base.train()
        print(colorful.bold_blue(f'# of model params : {count_parameters(base)} / original: 617,474'))

        return base

    def get_eval_model(self):
        self.base.eval()
        return self.base

    def get_eval_dataset(self):
        test_dataset = self.scheduler.test_datasets
        val_dataset = self.scheduler.val_datasets
        return test_dataset, val_dataset

    def learn(self, task_loader, epoch, t, step):

        for epc_i in tqdm.trange(epoch, desc='epoch', leave=False, colour='blue'):
            # @every epc update with filtered data.
            if self.config['online_filtering'] and self.config['regress_warm_up'] <= epc_i:
                task_loader = update_taskloader(
                    dataset = self.scheduler.datasets[self.config['data_name']],
                    epc_i = epc_i+1, # epc_i count differs by 1
                    config = self.config,
                    writer = self.writer,
                )

            if task_loader == None:
                # task_loader==None when no sample remains after filtering
                continue

            for batch in tqdm.tqdm(task_loader, desc='step ', leave=False, colour='green'):
                if self.config['classification']:
                    x, clf_y, clf_gt_y, y, gt_y, idx = batch
                    x, clf_y, clf_gt_y = x.to(self.device), clf_y[0].to(self.device), clf_gt_y[0].to(self.device)
                    clean_mask = clf_y == clf_gt_y
                else:
                    x, y, gt_y, idx = batch
                    clf_y, clf_gt_y = None, None
                    x, y, gt_y = x.to(self.device), y.to(self.device), gt_y.to(self.device)
                    # formatting y as (B, n) (from (B, ))
                    if len(y.shape) == 1 :
                        y = torch.unsqueeze(y, dim=1)
                        gt_y = torch.unsqueeze(gt_y, dim=1)
                    clean_mask = torch.all(y == gt_y, dim=1)

                self.base.zero_grad(set_to_none=True)

                # vanilla
                y_hat = self.base(x)

                loss = torch.sum(self.criterion(input=y_hat, target=y, gt_target=gt_y, reduction='none',
                                      clean_mask=clean_mask, idx=idx, epoch=epc_i, step=step),
                                 dim = 1) # loss -> (B, )

                mean_loss = torch.mean(loss)
                mean_loss.backward()

                if self.config['noise']['corrupt_p'] > 0.0:
                    # print wrong label loss and clean label loss separately.
                    num_clean = torch.sum(clean_mask)
                    num_noisy = clean_mask.size(0) - num_clean
                    if num_clean > 0:
                        clean_loss = torch.sum(loss * clean_mask) / num_clean
                        self.writer.add_scalar('train/clean_loss', clean_loss, step)
                    if num_noisy > 0:
                        noisy_loss = torch.sum(loss * ~clean_mask) / num_noisy
                        self.writer.add_scalar('train/noisy_loss', noisy_loss, step)

                if self.config['tensorboard']['grad']:
                    write_grad_norm_scalar(self.writer, self.base, step)

                self.base.optimizer.step()

                self.writer.add_scalar('train/loss', mean_loss, step)

                step += 1

            self.base.lr_scheduler.step()
            self.criterion.flush(epc_i)

            if (epc_i+1)%self.config['eval_every'] == 0 or epc_i == epoch:
                print(colorful.bold_white(f'eval epoch {epc_i}, step {step}').styled_string)
                self.scheduler.eval(self, self.writer, epc_i)

        # based on val's best epoch, return test's corresponding epoch's value.
        best_val = min(self.cval_history)
        best_val_epoch = self.cval_history.index(best_val)
        best_test_from_val = self.test_history[best_val_epoch]
        self.writer.add_scalar('final/val/val_mae', best_val, 0)
        self.writer.add_scalar('final/val/test_mae', best_test_from_val, 0)
        print(colorful.bold_green(f'final/val/val_mae: {best_val}'))
        print(colorful.bold_green(f'final/val/test_mae: {best_test_from_val}'))

        return step
