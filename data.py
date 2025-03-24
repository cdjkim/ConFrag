from abc import ABC, abstractmethod
from copy import deepcopy
import os
import time
import warnings
import colorful
import pandas as pd
import numpy as np
import h5py
import torch
import torchvision
import typing as ty

import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter
from PIL import Image, ImageFile
from pathlib import Path

from utils import inject_noise, unique_target_fragmentation, save_noise_checksum, \
    new_inject_noise, range_target_fragmentation, build_X, build_y


# =========
# Scheduler
# =========
class DataScheduler():
    def __init__(self, config, writer):
        self.config = config
        self.datasets = {}
        self.total_step = 0
        self.nval_datasets = {}
        self.cval_datasets = {}
        self.test_datasets = {}

        # Prepare datasets
        dataset_name = config['data_name']

        self.datasets[dataset_name] = DATASET[dataset_name](self.config, writer, split='train', noise=True)
        self.nval_datasets[dataset_name] = DATASET[dataset_name](self.config, writer, split='val', noise=True)
        self.cval_datasets[dataset_name] = DATASET[dataset_name](self.config, writer, split='val', noise=False)
        self.test_datasets[dataset_name] = DATASET[dataset_name](self.config, writer, split='test', noise=False)
        self.total_step += len(self.datasets[dataset_name]) // self.config['batch_size']

        self.task_datasets = []
        self.nval_task_datasets = []
        self.cval_task_datasets = []
        self.test_task_datasets = []

        dataset = ConcatDataset([self.datasets[dataset_name].subsets['all']])
        self.task_datasets.append((config['total_epochs'], dataset))

    def __iter__(self):
        for t_i, (epoch, task) in enumerate(self.task_datasets):
            print(colorful.bold_green('\nProgress to Task %d' % t_i).styled_string)
            collate_fn = task.datasets[0].dataset.collate_fn
            task_loader = DataLoader(task, batch_size=self.config['batch_size'],
                            num_workers=self.config['num_workers'],
                            collate_fn=collate_fn,
                            drop_last=False,
                            pin_memory=True, # better when training on GPU.
                            shuffle=True)

            yield task_loader, epoch, t_i

    def get_task(self, t):
        return self.task_datasets[t][1]

    def get_dataloader(self, dataset):
        collate_fn = dataset.dataset.datasets[0].dataset.collate_fn
        return DataLoader(dataset, self.config['batch_size'], shuffle=True, collate_fn=collate_fn)

    def __len__(self):
        return self.total_step

    def eval(self, model, writer, step):
        for nval_dataset in self.nval_datasets.values():
            nval_dataset.eval(model, writer, step, eval_title='nval')
        for cval_dataset in self.cval_datasets.values():
            cval_dataset.eval(model, writer, step, eval_title='cval')
        for test_dataset in self.test_datasets.values():
            test_dataset.eval(model, writer, step, eval_title='test')
        model.train()


class DataTaskScheduler():
    def __init__(self, config, writer):
        self.config = config
        self.schedule = config['data_schedule']
        self.datasets = {}
        self.test_datasets = {}
        self.nval_datasets = {}
        self.cval_datasets = {}
        self.total_step = 0

        # Prepare datasets
        for i, stage in enumerate(self.schedule): # e.g., list of [subsets, shuffle]
            for j, subset in enumerate(stage['subsets']):  # e.g, [['mnist', 0], ['mnist', 1]]
                dataset_name, subset_name = subset

                if dataset_name not in self.datasets:
                    print(dataset_name)
                    self.datasets[dataset_name] = DATASET[dataset_name](self.config, writer, 'train', noise=True)
                    self.test_datasets[dataset_name] = DATASET[dataset_name](self.config, writer, 'test', noise=False)
                    self.nval_datasets[dataset_name] = DATASET[dataset_name](self.config, writer, 'val', noise=True)
                    self.cval_datasets[dataset_name] = DATASET[dataset_name](self.config, writer, 'val', noise=False)

                # this
                self.total_step += len(self.datasets[dataset_name].subsets[subset_name]) \
                                   // self.config['batch_size']

        self.task_datasets = []
        self.test_task_datasets = []
        self.nval_task_datasets = []
        self.cval_task_datasets = []
        self.task_i_to_dim = {}
        print(colorful.bold_green(f'task composition').styled_string)
        for task_i, stage in enumerate(self.schedule):
            subsets = []
            nval_subsets = []
            cval_subsets = []
            test_subsets = []
            subset_names = [] # which subsets are in the task_i
            clf_y_dim = [] # dimension of clf_y for each subset
            task_epoch = config['task_epochs']
            for dataset_name, subset_name in stage['subsets']:
                subsets.append(self.datasets[dataset_name].subsets[subset_name])
                test_subsets.append(self.test_datasets[dataset_name].subsets[subset_name])
                nval_subsets.append(self.nval_datasets[dataset_name].subsets[subset_name])
                cval_subsets.append(self.cval_datasets[dataset_name].subsets[subset_name])

                subset_names.append(subset_name)
                clf_y_dim.append(self.datasets[dataset_name].clf_y_to_dim[subset_name])

            dataset = ConcatDataset(subsets)
            test_dataset = ConcatDataset(test_subsets)
            nval_dataset = ConcatDataset(nval_subsets)
            cval_dataset = ConcatDataset(cval_subsets)

            self.task_datasets.append((task_epoch, dataset))
            self.test_task_datasets.append(test_dataset)
            self.nval_task_datasets.append(nval_dataset)
            self.cval_task_datasets.append(cval_dataset)

            assert len(set(clf_y_dim)) == 1, f'at task_i {task_i} clf_y_dim should be identical {clf_y_dim}'
            self.task_i_to_dim[task_i] = clf_y_dim[0] # map task_i to clf_y_dim

            print(f'task {task_i}: classifies {subset_names}, clf_y_dim {self.task_i_to_dim[task_i]}')

    def __iter__(self):
        for t_i, (epoch, task) in enumerate(self.task_datasets):
            print(colorful.bold_green('\nProgress to Task %d' % t_i).styled_string)
            collate_fn = task.datasets[0].dataset.collate_fn
            task_loader = DataLoader(task, batch_size=self.config['batch_size'],
                            num_workers=self.config['num_workers'],
                            collate_fn=collate_fn,
                            drop_last=False,
                            pin_memory=True, # better when training on GPU.
                            shuffle=True)

            yield task_loader, epoch, t_i

    def __len__(self):
        return self.total_step

    def eval(self, model, writer, step, t, task_labels):
        for nval_dataset in self.nval_datasets.values():
            nval_dataset.eval(model, writer, step, t, task_labels, eval_title='nval')
        for cval_dataset in self.cval_datasets.values():
            cval_dataset.eval(model, writer, step, t, task_labels, eval_title='cval')
        for test_dataset in self.test_datasets.values():
            test_dataset.eval(model, writer, step, t, task_labels, eval_title='test')
        model.train()


# ================
# Generic Datasets
# ================
class BaseDataset(Dataset, ABC):
    name = 'base'
    dataset_size = 0

    def __init__(self, config, split):
        self.config = deepcopy(config)
        self.subsets = dict()
        self.split = split
        self.data = None

    def __len__(self):
        return self.dataset_size

    def eval(self, model, writer: SummaryWriter, step, t=None, task_labels=None, eval_title=''):
        if self.config['eval']:
            self._eval_model(model, writer, step, t, task_labels, eval_title)

    @abstractmethod
    def _eval_model(self, model, writer: SummaryWriter, step, eval_title):
        raise NotImplementedError

    def collate_fn(self, batch):
        return default_collate(batch)


class RegressionDataset(BaseDataset):
    targets = NotImplemented

    @torch.no_grad()
    def _eval_model(self, model, writer: SummaryWriter, cur_epc, t, task_labels, eval_title):

        if 'fragment' in self.config['model_name']:
            self._eval_model_fragment(model, writer, cur_epc, t, task_labels, eval_title)

        elif self.config['classification']:
            self._eval_model_classification(model, writer, cur_epc, eval_title)

        else:
            self._eval_model_regression(model, writer, cur_epc, eval_title)

    @torch.no_grad()
    def _eval_model_fragment(self, model, writer: SummaryWriter, cur_epc, t, task_labels, eval_title):
        model.eval()
        base = model.get_eval_model(t)
        test_dataset, clf_y_dim = model.get_eval_dataset(t)
        data = DataLoader(test_dataset, batch_size=self.config['eval_batch_size'],
                        num_workers=self.config['eval_num_workers'], collate_fn=self.collate_fn)

        if model.disc_type == 'classifier':
            total = 0.
            correct = 0.
            loss_l = []
            for x, clf_y, gt_clf_y, y, gt_y, info in iter(data):
                x = x.to(model.device)
                clf_y = clf_y[clf_y_dim]
                for new_lbl, (_, clf_lbl) in enumerate(task_labels):
                    clf_y[clf_y==clf_lbl] = new_lbl
                clf_y = clf_y.to(model.device)

                y_hat = base(x)
                pred = y_hat.argmax(dim=1)
                loss_l.append(model.criterion(input=y_hat, target=clf_y))
                total += y.size(0)
                correct += (pred == clf_y).float().sum()

            # Overall accuracy
            accuracy = correct / total
            mean_loss = sum(loss_l) / len(loss_l)
            writer.add_scalar(f'{eval_title}/accuracy/expert_{t}', accuracy, cur_epc)
            writer.add_scalar(f'{eval_title}/loss/expert_{t}', mean_loss, cur_epc)

            if eval_title in ['cval', 'test']:
                print(f'{eval_title}/accuracy/epc', accuracy)
                print(f'{eval_title}/loss/epc', mean_loss)

        else:
            targets = test_dataset.datasets[0].dataset.targets
            self._eval_regression_helper(data, model, targets, base, writer, eval_title, cur_epc, t=t)

        writer.flush()
        base.train()

    @torch.no_grad()
    def _eval_model_classification(self, model, writer: SummaryWriter, cur_epc, eval_title='eval'):
        model.eval()
        base = model.get_eval_model()

        eval_dataset = self
        data = DataLoader(eval_dataset, batch_size=self.config['eval_batch_size'],
                          num_workers=self.config['eval_num_workers'], collate_fn=self.collate_fn)
        total = 0.
        correct = 0.
        loss_l = []
        for x, clf_y, gt_clf_y, y, gt_y, info in iter(data):
            x, clf_y = x.to(model.device), clf_y[0].to(model.device)
            y_hat = base(x)
            pred = y_hat.argmax(dim=1)
            loss_l.append(model.criterion(input=y_hat, target=clf_y))
            total += y.size(0)
            correct += (pred == clf_y).float().sum()

        # Overall accuracy
        accuracy = correct / total
        mean_loss = sum(loss_l) / len(loss_l)
        writer.add_scalar(f'{eval_title}/accuracy', accuracy, cur_epc)
        writer.add_scalar(f'{eval_title}/loss', mean_loss, cur_epc)

        writer.flush()
        base.train()

    @torch.no_grad()
    def _eval_model_regression(self, model, writer: SummaryWriter, cur_epc, eval_title):
        model.eval()
        base = model.get_eval_model()

        eval_dataset = self

        data = DataLoader(eval_dataset, batch_size=self.config['eval_batch_size'],
            num_workers=self.config['eval_num_workers'], collate_fn=self.collate_fn)

        self._eval_regression_helper(data, model, eval_dataset.targets, base, writer, eval_title, cur_epc)
        base.train()


    def _eval_regression_helper(self, data, model, targets, base, writer, eval_title, cur_epc, t=None):

        if len(targets.shape) == 1:
            avg_error = torch.zeros(1).to(model.device)
            avg_perc_error = torch.zeros(1).to(model.device)
        else:
            avg_error = torch.zeros(targets.shape[1]).to(model.device)
            avg_perc_error = torch.zeros(targets.shape[1]).to(model.device)

        total = 0.
        losses = []
        loss_angular = 0.
        for get_item_out in iter(data):
            if self.config['model_name'] == 'garg' and self.config['data_name'] == 'msd':
                x, y, gt_y, info, y_cls = get_item_out
                with torch.no_grad():
                    if len(y.shape) == 1 :
                        y = torch.unsqueeze(y, dim=1)
                        gt_y = torch.unsqueeze(gt_y, dim=1)
                        y_cls = torch.unsqueeze(y_cls, dim=1)
                    x, y = x.to(model.device), y.to(model.device)
                    y_cls = y_cls.to(model.device)

                    y_hat = base(x)
                    loss = model.criterion(y_hat, y_cls, test=True)

                    y_hat = y_hat.view(y_hat.shape[0], -1, 2)
                    y_hat = torch.sum(torch.argmax(y_hat, dim=2), dim=1) + model.min_y
                    new_y_hat = []
                    for i in range(len(y_hat)):
                        new_y_hat.append(self.c_to_lbl[int(y_hat[i])])
                    y_hat = torch.Tensor(new_y_hat).to(model.device)
                    y_hat = torch.unsqueeze(y_hat, dim=1)

            else:
                if 'fragment' in self.config['model_name']:
                    x, clf_y, gt_clf_y, y, gt_y, info = get_item_out
                else:
                    x, y, gt_y, info = get_item_out
                with torch.no_grad():
                    if len(y.shape) == 1 :
                        y = torch.unsqueeze(y, dim=1)
                        gt_y = torch.unsqueeze(gt_y, dim=1)
                    x, y = x.to(model.device), y.to(model.device)

                    y_hat = base(x)
                    loss = model.criterion(input=y_hat, target=y, test=True)

                    if self.config['model_name'] == 'garg':
                        y_hat = y_hat.view(y_hat.shape[0], -1, 2)
                        y_hat = torch.sum(torch.argmax(y_hat, dim=2), dim=1) + model.min_y
                        y_hat = torch.unsqueeze(y_hat, dim=1)

            # restoration required for timeseries dataset
            y = self.restore(y, info)
            y_hat = self.restore(y_hat, info)

            total += y.size(0)

            # MAE (Mean Absolute Error)
            # 1 / N * sum(abs(y-y_hat))
            avg_error = avg_error + torch.sum(abs(y-y_hat), dim=0)

            # MAPE (Mean Absolute Percentage Error)
            # 1 / N * sum(abs(1-y_hat/y))
            avg_perc_error = avg_perc_error + torch.sum(abs(1- y_hat/y), dim=0)
            #avg_perc_error += torch.sum(abs(1 - y_hat/y)).item()

            losses.append(loss)

        mae = (torch.sum(avg_error) / total).item()
#        mape = (torch.sum(avg_perc_error) / total).item()
        mean_loss = (sum(losses) / len(losses)).item()

        postfix = 'epc' if t == None else f'expert_{t}'
        writer.add_scalar(f'{eval_title}/mae/{postfix}', mae, cur_epc)
#        writer.add_scalar(f'{eval_title}/mape/epc', mape, cur_epc)
        writer.add_scalar(f'{eval_title}/loss/{postfix}', mean_loss, cur_epc)

        if eval_title in ['cval', 'test']:
            print(f'{eval_title}/mae/{postfix}', mae)
#            print(f'{eval_title}/mape/{postfix}', mape)
            print(f'{eval_title}/loss/{postfix}', mean_loss)

        if 'fragment' not in self.config['model_name']:
            if eval_title == 'test':
                model.test_history.append(mae)
            elif eval_title == 'nval':
                model.nval_history.append(mae)
            elif eval_title == 'cval':
                model.cval_history.append(mae)

        writer.flush()

    def restore(self, value, info):
        return value

    def _preprocess(self, split, noise):
        self.org_targets = deepcopy(self.targets)
        if 'noise' in self.config and noise:
            if self.config['data_name'] in ['shift15m']:
                self.targets, noise_rate = new_inject_noise(self.targets, self.config['noise'])
                if split == 'train':
                    self.config['noise']['corrupt_p'] = noise_rate
            else:
                self.targets, noise_rate = inject_noise(self.targets, self.config['noise'])
                if split == 'train':
                    self.config['noise']['corrupt_p'] = noise_rate

        self.org_crpt_targets = deepcopy(self.targets)

        if 'fragment' in self.config['model_name'] or \
            'spr' in self.config['model_name'] or \
                self.config['classification'] :

            if self.config['data_name'] == 'shift15m':
                self.clf_targets, self.gt_clf_targets, self.split_map = range_target_fragmentation(
                    self.targets, self.org_targets, self.config['label_split'], split)
            else:
                self.clf_targets, self.gt_clf_targets, self.split_map = unique_target_fragmentation(
                    self.targets, self.org_targets, self.config['label_split'], split)

        ckpt_dir = os.path.join(self.config['log_dir'], 'ckpts')
        save_noise_checksum(ckpt_dir, self.targets)

    def update_data(self, idcs, ignore_idcs=False):
        if ignore_idcs:
            self.total_idcs = list(set(self.total_idcs) - set(idcs))
        else:
            self.total_idcs = idcs

        self.subsets['all'] = Subset(self, self.total_idcs)
        self.dataset_size = len(self.total_idcs)

    def reset_targets(self):
        self.targets = deepcopy(self.org_crpt_targets)
        self.total_idcs = list(range(0, len(self.data)))
        self.subsets['all'] = Subset(self, self.total_idcs)
        self.dataset_size = len(self.total_idcs)

    def update_fragmentation(self, label_coverage, split):
        """ update the clf targets using continuous -> discrete mapping.
        Args:
            label_coverage(float) : coverage of full label space.
            split(str) : split name (train, test, val)
        Returns:
            None
        """
        if self.config['data_name'] == 'shift15m':
            self.clf_targets, self.gt_clf_targets, self.split_map = range_target_fragmentation(
                self.targets, self.org_targets, self.config['label_split'], split,
                label_coverage=label_coverage)
        else:
            self.clf_targets, self.gt_clf_targets, self.split_map = unique_target_fragmentation(
                self.targets, self.org_targets, self.config['label_split'], split,
                label_coverage=label_coverage)

    def jitter_data(self, task_i, task_info, label_coverage):
        """ randomly jitter the range of the splits.
        Args:
            t(int) : task_id
            task_info(dict) : task_info
            label_coverage(float) : coverage of full label space.
        Returns:
            jittered task dataloader
        """
        self.update_fragmentation(label_coverage, split="train")

        subsets = {}
        dataset_size = 0
        clf_y_to_dim = {}
        if 'fragment' in self.config['model_name']:
            for dim_i, splits in self.split_map.items():
                for y, _ in splits.items(): # splits -> {clf_id: split_range}
                    subsets[y] = Subset(
                        self, np.argwhere(np.array(self.clf_targets[dim_i]) == y).squeeze()
                    )
                    clf_y_to_dim[y] = dim_i
                    dataset_size += len(subsets[y])
                    print(f'split y={y}: dim {dim_i}, len {len(subsets[y])}')
            # clf_targets must be updated, since the idcs are obtained this way.

        task_i_to_dim = {}
        print(colorful.bold_green(f'task composition').styled_string)

        task_subsets = []
        task_subset_names = [] # which subsets are in the task_i
        clf_y_dim = [] # dimension of clf_y for each subset

        for _, subset_name in task_info:
            task_subsets.append(subsets[subset_name])
            task_subset_names.append(subset_name)
            clf_y_dim.append(clf_y_to_dim[subset_name])

        task_dataset = ConcatDataset(task_subsets)
        assert len(set(clf_y_dim)) == 1, f'at task_i {task_i} clf_y_dim should be identical {clf_y_dim}'
        task_i_to_dim[task_i] = clf_y_dim[0] # map task_i to clf_y_dim

        print(f'task {task_i}: classifies {task_subset_names}, clf_y_dim {task_i_to_dim[task_i]}')

        task_loader = DataLoader(task_dataset, batch_size=self.config['batch_size'],
                        num_workers=self.config['num_workers'],
                        collate_fn=self.collate_fn,
                        drop_last=False,
                        pin_memory=True, # better when training on GPU.
                        shuffle=True)

        return task_loader


class ClassificationDataset(BaseDataset):
    num_classes = NotImplemented
    targets = NotImplemented

    def _eval_model(self, model, writer: SummaryWriter, step, eval_title):
        model = model.get_finetuned_model()

        totals = []
        corrects = []
        for subset_name, subset in self.subsets.items():
            data = DataLoader(subset, batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'], collate_fn=self.collate_fn)
            total = 0.
            correct = 0.
            for x, y in iter(data):
                with torch.no_grad():
                    x, y = x.to(model.device), y.to(model.device)
                    pred = model(x).argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).float().sum()

            totals.append(total)
            corrects.append(correct)
            accuracy = correct / total
            writer.add_scalar('accuracy/%s/%s/%s' % (eval_title, self.name, subset_name),
                              accuracy, step)
            print('accuracy/%s/%s/%s' % (eval_title, self.name, subset_name), accuracy)

        # Overall accuracy
        total = sum(totals)
        correct = sum(corrects)
        accuracy = correct / total
        writer.add_scalar('accuracy/%s/%s/overall' % (eval_title, self.name),
                        accuracy, step)

        print('accuracy/%s/%s/overall' % (eval_title, self.name), accuracy)
        writer.flush()


# =================
# Concrete Datasets
# =================
class AFAD(RegressionDataset):
    name = 'afad'

    def __init__(self, config, writer, split, noise=False):
        super(AFAD, self).__init__(config, split)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.config['x_h'] == 128:
            self.data_folder = '128x128'
            transform = []
        else:
            self.data_folder = 'data'
            transform = [transforms.Resize((config['x_h'], config['x_w']))]

        self.transform_list = {}
        self.transform_list['train'] = transforms.Compose([
            *transform,
            # lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.RandomCrop(config['x_h'], padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_list['eval'] = transforms.Compose([
            *transform,
            # lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.writer = writer
        self.transform = self.transform_list['train' if split == 'train' else 'eval']
        self.idx_to_realname = list()
        file_path = os.path.join(config['data_root'], self.name, 'afad_balanced_1251.csv')

        print(colorful.bold_green(f'reading from file path: {split} {file_path}').styled_string)
        df = pd.read_csv(file_path)

        self.data_paths = np.array(df['path'])
        self.targets = np.array(df['age'])

        # Save range of target
        self.max_target = np.max(self.targets)
        self.min_target = np.min(self.targets)
        self.range_target = self.max_target - self.min_target

        split_idx = np.where(np.array(df['split']) == split)
        self.data_paths = self.data_paths[split_idx]
        self.targets = self.targets[split_idx]
        self.data = []
        for data_path in self.data_paths:
            self.data.append(data_path)
        assert len(self.data) == len(self.targets)
        self.total_idcs = list(range(0, len(self.data)))

        # inject noise, generate clf_targets, filter dataset
        self._preprocess(split, noise)

        print(colorful.bold_coral(f'{split} len(self.data): {len(self.data)}').styled_string)
        print(colorful.bold_coral(f'{split} minimum_age: {self.targets.min()}').styled_string)
        print(colorful.bold_coral(f'{split} maximum_age: {self.targets.max()}').styled_string)

        self.clf_y_to_dim = {}
        if 'fragment' in config['model_name']:
            for dim_i, splits in self.split_map.items():
                for y, _ in splits.items():
                    self.subsets[y] = Subset(
                        self, np.argwhere(np.array(self.clf_targets[dim_i]) == y).squeeze()
                    )
                    self.clf_y_to_dim[y] = dim_i
                    self.dataset_size += len(self.subsets[y])
                    print(f'split y={y}: dim {dim_i}, len {len(self.subsets[y])}')
        else:
            self.subsets['all'] = Subset(self, self.total_idcs)
            self.dataset_size = len(self.total_idcs)
        print('dataset error', np.mean(abs(self.targets - self.org_targets)))


    def set_transform(self, phase):
        assert phase in ['train', 'eval']
        self.transform = self.transform_list[phase]

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = self.transform(Image.open(os.path.join(self.config['data_root'],
                                                       self.name, self.data_folder, self.data[idx])))
        if self.config['classification']:
            clf_y = []
            gt_clf_y = []
            for i, _ in enumerate(self.config['label_split']):
                clf_y.append(self.clf_targets[i][idx])
                gt_clf_y.append(self.gt_clf_targets[i][idx])

        gt_y = self.org_targets[idx].astype(np.float32)
        y = self.targets[idx].astype(np.float32)

        if self.config['classification']:
            return x, clf_y, gt_clf_y, y, gt_y, idx

        return x, y, gt_y, idx


class IMDB_WIKI(RegressionDataset):
    name = 'imdb_wiki'

    def __init__(self, config, writer, split, noise=False):
        super(IMDB_WIKI, self).__init__(config, split)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.config['x_h'] == 128:
            self.data_folder = '128x128'
            transform = []
        else:
            self.data_folder = '.'
            transform = [transforms.Resize((config['x_h'], config['x_w']))]

        self.transform_list = {}
        self.transform_list['train'] = transforms.Compose([
            *transform,
            # lambda x: x.convert('RGB'),
            transforms.RandomCrop(config['x_h'], padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_list['eval'] = transforms.Compose([
            *transform,
            # lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.writer = writer
        self.transform = self.transform_list['train' if split == 'train' else 'eval']

        self.idx_to_realname = list()
        self.name = 'imdb_wiki_clean'
        file_path = os.path.join(config['data_root'], self.name, 'imdb_clean_balanced_cnt1000.csv')

        print(colorful.bold_green(f'reading from file path: {split} {file_path}').styled_string)
        df = pd.read_csv(file_path)

        self.data = np.array(df['path'])
        self.targets = np.array(df['age'])

        # Save range of target
        self.max_target = np.max(self.targets)
        self.min_target = np.min(self.targets)
        self.range_target = self.max_target - self.min_target

        if 'debug' in self.config.keys() and self.config['debug'] and split == 'train':
            split_idx = np.where(np.array(df['split']) == 'val')
        else:
            split_idx = np.where(np.array(df['split']) == split)

        self.data = self.data[split_idx]
        self.targets = self.targets[split_idx]

        assert len(self.data) == len(self.targets)
        total_idcs = list(range(0, len(self.data)))

        # inject noise, generate clf_targets, filter dataset
        self._preprocess(split, noise)

        self.clf_y_to_dim = {}
        if 'fragment' in config['model_name']:
            for dim_i, splits in self.split_map.items():
                for y, _ in splits.items(): # splits -> {clf_id: split_range}
                    self.subsets[y] = Subset(
                        self, np.argwhere(np.array(self.clf_targets[dim_i]) == y).squeeze()
                    )
                    self.clf_y_to_dim[y] = dim_i
                    self.dataset_size += len(self.subsets[y])
                    print(f'split y={y}: dim {dim_i}, len {len(self.subsets[y])}')
            # clf_targets must be updated, since the idcs are obtained this way.
        else:
            self.subsets['all'] = Subset(self, total_idcs)
            self.dataset_size = len(total_idcs)
        print('dataset error', np.mean(abs(self.targets - self.org_targets)))

    def set_transform(self, phase):
        assert phase in ['train', 'eval']
        self.transform = self.transform_list[phase]

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pil_x = Image.open(os.path.join(self.config['data_root'], self.name, self.data_folder, self.data[idx]))
            x = self.transform(pil_x)

        if self.config['classification']:
            clf_y = []
            gt_clf_y = []
            for i, _ in enumerate(self.config['label_split']):
                clf_y.append(self.clf_targets[i][idx])
                gt_clf_y.append(self.gt_clf_targets[i][idx])

        gt_y = self.org_targets[idx].astype(np.float32)
        y = self.targets[idx].astype(np.float32)

        if self.config['classification']:
            return x, clf_y, gt_clf_y, y, gt_y, idx

        return x, y, gt_y, idx


class MSD(RegressionDataset):
    """ Million Song Dataset
    """
    name = 'msd'

    def __init__(self, config, writer, split, noise=False):
        super(MSD, self).__init__(config, split)

        self.writer = writer

        self.idx_to_realname = list()
        file_path = os.path.join(config['data_root'], self.name, 'bal550')


        print(colorful.bold_green(f'reading from file path: {split} {file_path}').styled_string)
        # df = pd.read_csv(file_path)

        dir_ = Path(file_path)
        def load(item):
            return {
                x: ty.cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy'))  # type: ignore[code]
                for x in ['train', 'val', 'test']
            }

        N_data = load('N') if dir_.joinpath('N_train.npy').exists() else None
        y_data = load('y')

        X = build_X(
            N=N_data,
            C=[],
            data_path=dir_,
            normalization='quantile',
            num_nan_policy='mean',
            cat_nan_policy='new',
            cat_policy='indices',
            cat_min_frequency='cat_min_frequency',
            seed=self.config['random_seed'],
        )

        Y, y_info = build_y(y_data, 'mean_std')

        self.data = X[split]
        self.targets = Y[split]

        self.config['d_in'] = self.data.shape[1]

        # Save range of target
        self.max_target = np.max(self.targets)
        self.min_target = np.min(self.targets)
        self.range_target = self.max_target - self.min_target

        assert len(self.data) == len(self.targets)
        self.total_idcs = list(range(0, len(self.data)))

        # inject noise, generate clf_targets, filter dataset
        self._preprocess(split, noise)

        self.clf_y_to_dim = {}
        if 'fragment' in config['model_name']:
            for dim_i, splits in self.split_map.items():
                for y, _ in splits.items():
                    self.subsets[y] = Subset(
                        self, np.argwhere(np.array(self.clf_targets[dim_i]) == y).squeeze()
                    )
                    self.clf_y_to_dim[y] = dim_i
                    self.dataset_size += len(self.subsets[y])
                    print(f'split y={y}: dim {dim_i}, len {len(self.subsets[y])}')
        else:
            self.subsets['all'] = Subset(self, self.total_idcs)
            self.dataset_size = len(self.total_idcs)

        label_space = np.unique(self.targets)
        self.targets_cls = np.zeros(len(self.targets))
        self.org_targets_cls = np.zeros(len(self.targets))
        self.c_to_lbl = {}
        for c, lbl in enumerate(label_space):
            self.targets_cls[lbl == self.targets] = c
            self.org_targets_cls[lbl == self.org_targets] = c
            self.c_to_lbl[c] = lbl
        print('dataset error', np.mean(abs(self.targets - self.org_targets)))

    def set_transform(self, phase):
        assert phase in ['train', 'eval']
        self.transform = []

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = torch.from_numpy(self.data[idx])
        if self.config['classification']:
            clf_y = []
            gt_clf_y = []
            for i, _ in enumerate(self.config['label_split']):
                clf_y.append(self.clf_targets[i][idx])
                gt_clf_y.append(self.gt_clf_targets[i][idx])

        gt_y = self.org_targets[idx].astype(np.float32)
        y = self.targets[idx].astype(np.float32)

        if self.config['classification']:
            return x, clf_y, gt_clf_y, y, gt_y, idx

        if self.config['model_name'] == 'garg':
            y_cls = self.targets_cls[idx].astype(np.float32)
            return x, y, gt_y, idx, y_cls

        return x, y, gt_y, idx


class SHIFT15M(RegressionDataset):
    name = 'shift15m'

    def __init__(self, config, writer, split='train', noise=False):
        super(SHIFT15M, self).__init__(config, split)

        self.writer = writer
        self.transform = torchvision.transforms.Compose([])

        root_path = os.path.join(config['data_root'], self.name, 'data')
        file_name_base = 'item_price_bal_2000_16084_40000'

        csv_file_path = os.path.join(root_path, f'{file_name_base}_{split}.csv')
        h5_file_path = os.path.join(root_path, f'{file_name_base}_{split}.h5')

        df = pd.read_csv(csv_file_path)
        print(colorful.bold_green(f'reading from file path: {csv_file_path}').styled_string)
        f = h5py.File(h5_file_path, 'r')
        print(colorful.bold_green(f'reading from file path: {h5_file_path}').styled_string)

        self.data = f['feature']
        self.targets = np.expand_dims(np.array(df['price']), axis=1)
        self.targets = self.targets / 1000 # approximate yen/dollar exchange rate

        # Save range of target
        self.max_target = np.max(self.targets)
        self.min_target = np.min(self.targets)
        self.range_target = self.max_target - self.min_target

        assert self.data.shape[0] == self.targets.shape[0]
        self.total_idcs = list(range(0, len(self.data)))

        self._preprocess(split, noise)

        num_noise = np.sum(np.any(self.targets != self.org_targets, axis=1))
        noise_rate = num_noise / len(self.targets)
        print(colorful.bold_coral(f'{split} self.data.shape: {self.data.shape}').styled_string)
        print(colorful.bold_coral(f'number of noisy data: {num_noise}').styled_string)

        self.targets = self.targets.reshape(-1)
        self.org_targets = self.org_targets.reshape(-1)
        self.org_crpt_targets = self.org_crpt_targets.reshape(-1)

        self.clf_y_to_dim = {}
        if 'fragment' in config['model_name']:
            for dim_i, splits in self.split_map.items():
                for y, _ in splits.items():
                    self.subsets[y] = Subset(
                        self, np.argwhere(np.array(self.clf_targets[dim_i]) == y).squeeze()
                    )
                    self.clf_y_to_dim[y] = dim_i
                    self.dataset_size += len(self.subsets[y])
                    print(f'split y={y}: dim {dim_i}, len {len(self.subsets[y])}')
        else:
            self.subsets['all'] = Subset(self, self.total_idcs)
            self.dataset_size = len(self.total_idcs)
        print('dataset error', np.mean(abs(self.targets - self.org_targets)))

    def set_transform(self, phase):
        assert phase in ['train', 'eval']
        self.transform = self.transform

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):

        x = torch.Tensor(self.data[idx]).to(torch.float32)
        y = self.targets[idx].astype(np.float32)
        gt_y = self.org_targets[idx].astype(np.float32)

        if self.config['classification']:
            clf_y = []
            gt_clf_y = []
            for i, _ in enumerate(self.config['label_split']):
                clf_y.append(self.clf_targets[i][idx])
                gt_clf_y.append(self.gt_clf_targets[i][idx])

            return x, clf_y, gt_clf_y, y, gt_y, idx

        return x, y, gt_y, idx


DATASET = {
    IMDB_WIKI.name: IMDB_WIKI,
    SHIFT15M.name: SHIFT15M,
    AFAD.name: AFAD,
    MSD.name: MSD,
}
