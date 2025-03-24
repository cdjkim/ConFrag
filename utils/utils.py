from typing import Tuple

import logging
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import kornia
from colorlog import ColoredFormatter
import os
import sys
import colorful

from sklearn.neighbors import KernelDensity
import tqdm


def setup_logger():
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

def override_config(config, override):
    # Override options
    for option in override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)
    return config

class SelfSupTransform():
    def __init__(self, image_shape):
        transform = [
            kornia.augmentation.RandomResizedCrop(size=image_shape[:2]),
            kornia.augmentation.RandomHorizontalFlip()]
        if image_shape[2] == 3:
            transform.append(kornia.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.2, p=0.5))
        self.transform = transforms.Compose(transform)
    def __call__(self, image):
        return self.transform(image)

def mixup(x, y, alpha=1.0, use_cuda=True):
    '''
    Perform within class mixup
    Returns mixed inputs
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x_lst = []
    y_lst = []
    for unq in torch.unique(y):
        clf_mask = y == unq
        clf_sum = torch.sum(clf_mask)
        ind_shuffled = torch.randperm(clf_sum)

        mixed_x = lam * x[clf_mask] + (1 - lam) * x[clf_mask][ind_shuffled, :]
        mixed_x_lst.append(mixed_x)
        y_lst.append(torch.ones(clf_sum, dtype=torch.int64) * unq)

    ind_shuffled = torch.randperm(clf_mask.shape[0])
    mixed_x = torch.cat(mixed_x_lst)[ind_shuffled]
    y = torch.cat(y_lst)[ind_shuffled]

    return mixed_x, y

# C-mixup methods
def cmixup_sampling_probs(config, dataset):
    ''' generate/save/load sampling_probs matrix
    Parameters:
      - config: config
      - dataset: dataset

    Return:
        sampling_probs: (N, N) matrix
        dataset: dataset
        targets: targets
    '''
    bandwidth = config['cmixup']['kde_bandwidth']
    beta_alpha = config['cmixup']['beta_alpha']
    manifold = config['cmixup']['manifold_mixup']

    # organize sampling_probs matrix file_name
    logdir = config['log_dir']
    mname = config['model_name']
    dname = config['data_name']
    dver = config['data_version']
    ntype = config['noise']['type']
    crrp = config['noise']['corrupt_p']
    rs = config['random_seed']
    if mname in logdir.split('/'):
        logdir = logdir[:logdir.find(mname)]
        file_name = f'{logdir}{mname}/{dname}_{dver}_{ntype}_crrP-{crrp}_rs-{rs}_bandwidth-{bandwidth}.npy'
    else:
        file_name = f'{logdir}/{dname}_{dver}_{ntype}_crrP-{crrp}_rs-{rs}_bandwidth-{bandwidth}.npy'
    print(colorful.bold_green(f'matrix file name: {file_name}'))

    targets = dataset.targets # ndarray of target
    num_train = targets.shape[0]
    targets = targets.reshape(num_train, -1)
    all_index = range(num_train)

    if os.path.exists(file_name):
        print(colorful.bold_green(f'Load matrix from {file_name}'))
        sampling_probs = np.load(file_name)
    else:
        print(colorful.bold_green(f'Generate matrix'))
        sampling_probs = np.ones((num_train, num_train), dtype=np.float16)
        if config['debug']:
            sampling_probs /= num_train
        else:
            for i in tqdm.trange(num_train, desc=f'sampling_prob_matrix', colour='green'):
                target_i = targets[i]
                # For N-D label space, we use euclidean distance
                distance = np.sum((targets - target_i)**2, axis=1, keepdims=True)**0.5
                # distance to self_target is 0
                kd = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(np.array([[0]]))
                sampling_prob_i = np.exp(kd.score_samples(distance))
                sampling_prob_i /= np.sum(sampling_prob_i)
                sampling_probs[i] = sampling_prob_i
            sampling_probs = np.array(sampling_probs, dtype=np.float16)
            print(colorful.bold_green(f'Save matrix from {file_name}'))
            np.save(file_name, sampling_probs)

    print(colorful.bold_green(f'matrix_size:{sys.getsizeof(sampling_probs)/1024/1024/1024:.5}'))

    return sampling_probs, dataset, targets

def cmixup_sampling_probs_batch(config, y):
    ''' generate (batch)sampling_probs matrix
    Parameters:
      - config: config['cmixup']
      - y: y of minibatch

    Return:
        sampling_probs: (B, B) matrix
    '''
    bandwidth = config['kde_bandwidth']
    beta_alpha = config['beta_alpha']
    manifold = config['manifold_mixup']

    targets = y.cpu().numpy()
    sampling_probs = np.ones((targets.shape[0], targets.shape[0]), dtype=np.float16)
    for i in range(targets.shape[0]):
        target_i = targets[i]
        # For N-D label space, we use euclidean distance
        distance = np.sum((targets - target_i)**2, axis=1, keepdims=True)**0.5
        # distance to self_target is 0
        kd = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(np.array([[0]]))
        sampling_prob_i = np.exp(kd.score_samples(distance))
        sampling_prob_i /= np.sum(sampling_prob_i)
        sampling_probs[i] = sampling_prob_i
    sampling_probs = np.array(sampling_probs, dtype=np.float16)

    return sampling_probs

def cmixup_sampling(config, sampling_probs, dataset, x, y, idx):
    ''' return x_j, y_j based on sampling_probs
    Parameters:
      - config: config (not config['cmixup'])
      - sampling_probs
      - datatset
      - x: x of batch
      - y: y of batch
      - idx: dataset index of batch

    Return:
      - x_j, y_j : to be interpolated data
    '''

    if config['cmixup']['batch_cmixup']:
        sampling_probs = cmixup_sampling_probs_batch(config['cmixup'], y)
        base_idx = np.arange(y.shape[0])

    else:
        base_idx = idx

    # sampling idx_j
    num_sample = sampling_probs.shape[0]
    idx_j = np.array([np.random.choice(np.arange(num_sample), p = sampling_probs[self_idx])
                        for self_idx in base_idx])


    # sampling x_j, y_j
    # batch_cmixup
    if config['cmixup']['batch_cmixup']:
        x_j = x[idx_j]
        y_j = y[idx_j]

    else:
        # original_cmixup
        # when multi-indexing on dataset is impossible (due to image load)
        if config['data_name'] in ['agedb', 'imdb_wiki', 'utkface']:
            x_j = torch.zeros(x.shape)
            y_j = np.zeros(y.shape)
            for i, index in enumerate(idx_j):
                x_j[i], y_j[i], _, _ = dataset[index]
            x_j = x_j.to(config['device'])
            y_j = torch.Tensor(y_j).to(config['device'])

        # when multi-indexing on dataset is possible
        else:
            x_j = torch.Tensor(dataset.data[idx_j].astype(np.float32)).to(config['device'])
            y_j = torch.Tensor(dataset.targets[idx_j].astype(np.float32)).to(config['device'])

    return x_j, y_j

def cmixup_forward(config, model, x, y, x_j, y_j):
    ''' return x_j, y_j based on sampling_probs
    Parameters:
        - config: config['cmixup']

    Return:
        - mix_x: mixed data
        - mix_y: mixed target
        - y_hat: prediction of mix_x
    '''
    lambd = np.random.beta(config['beta_alpha'], config['beta_alpha'])
    if config['manifold_mixup']:
        # extract features and mixup it
        _, x = model(x, return_feature=True)
        _, x_j = model(x_j, return_feature=True)

        mix_x = (x * lambd) + (x_j * (1-lambd))
        mix_y = (y * lambd) + (y_j * (1-lambd))

        y_hat = model.fc(mix_x)
    else:
        mix_x = (x * lambd) + (x_j * (1-lambd))
        mix_y = (y * lambd) + (y_j * (1-lambd))

        y_hat = model(mix_x)

    return mix_x, mix_y, y_hat


class NTXentLoss(nn.Module):
    """This code is based on the https://github.com/chagmgang/simclr_pytorch/blob/master/nt_xent_loss.py"""
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def write_grad_norm_scalar(writer, component, step):
    """
    Pytorch grad_norm implementation:
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
    ┊ for p in parameters]))
    """
    total_norm = 0
    parameters = [p for p in component.parameters()
                if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar('train/grad_norm/step', total_norm, step)


def get_exp_path(exp_path, dir_name, use_recent_identifier=False):
    if use_recent_identifier == False:
        # if use_recent_identifier==False, identifier should be passed by exp_path
        return os.path.join(exp_path, dir_name)

    else:
        identifier_list = os.listdir(exp_path)

        if len(identifier_list) == 0:
            raise ValueError(f'Exp path does not exist at {exp_path}')

        recent_identifier = 0
        for identifier in identifier_list:
            if not identifier.isnumeric():
                print(colorful.red(f"Unexpected identifier found at {exp_path}/{identifier}").styled_string)
                continue

            # choose the most recent identifier
            if int(identifier) > recent_identifier:
                recent_identifier = int(identifier)

        if recent_identifier == 0:
            raise FileNotFoundError(f'Cannot find valid identifier at {exp_path}')

        if len(identifier_list) > 1:
            print(colorful.red(f"Found {len(identifier_list)} identifiers at {exp_path}").styled_string)
            print(colorful.red(f"The most recent one is chosen: {recent_identifier}").styled_string)

        return_path = os.path.join(exp_path, str(recent_identifier), dir_name)

        if not os.path.exists(return_path):
            raise FileNotFoundError(f'Directory does not exist at {return_path}')

        return return_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)