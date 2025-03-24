from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NTXentLoss, SelfSupTransform


class Component(nn.Module, ABC):
    def __init__(self, config, feature_extractor: nn.Module):
        super().__init__()
        self.config = config
        self.device = config['device'] if 'device' in config else 'cuda'
        self.feature = feature_extractor

        self.optimizer = NotImplemented
        self.lr_scheduler = NotImplemented

    @abstractmethod
    def forward(self, x):
        pass

    def setup_optimizer(self, optim_config):
        self.optimizer = getattr(torch.optim, optim_config['type'])(
            self.parameters(), **optim_config['options'])

    def setup_lr_scheduler(self, lr_config):
        self.lr_scheduler = getattr(torch.optim.lr_scheduler, lr_config['type'])(
            self.optimizer, **lr_config['options'])

    def _clip_grad_value(self, clip_value):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_value_(group['params'], clip_value)

    def _clip_grad_norm(self, max_norm, norm_type=2):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    def clip_grad(self):
        clip_grad_config = self.config['clip_grad']
        if clip_grad_config['type'] == 'value':
            self._clip_grad_value(**clip_grad_config['options'])
        elif clip_grad_config['type'] == 'norm':
            self._clip_grad_norm(**clip_grad_config['options'])
        elif clip_grad_config['type'] == 'none':
            pass
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(clip_grad_config.type))


class Classify(Component):
    def __init__(self, config, feature_extractor, fc_in_dim, nb_classes):
        super(Classify, self).__init__(config, feature_extractor)

        self.fc = nn.Linear(fc_in_dim, nb_classes)
        self.to(self.device)

    def forward(self, x, return_feature=False):
        feature = self.feature(x)
        y_hat = self.fc(feature)
        if return_feature:
            return y_hat, feature
        return y_hat


class Regress(Component):
    def __init__(self, config, feature_extractor, fc_in_dim):
        super(Regress, self).__init__(config, feature_extractor)

        if 'num_head' not in config.keys():
            config['num_head'] = 1

        self.fc = nn.Linear(fc_in_dim, config['num_head'])
        self.to(self.device)

    def forward(self, x, return_feature=False):
        feature = self.feature(x)
        y_hat = self.fc(feature)
        if return_feature:
            return y_hat, feature
        return y_hat
