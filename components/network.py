from copy import deepcopy

from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod
from components.component import Regress, Classify, Component
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50
from torch import Tensor



class ResNet(Classify):
    def __init__(self, config, nb_classes):
        if 'pretrained' in config.keys() and config['pretrained'] == True:
            weights = 'DEFAULT'
        else:
            weights = None

        if config['net'] == 'resnet50':
            resnet = resnet50(weights = weights)
        elif config['net'] == 'resnet34':
            resnet = resnet34(weights = weights)
        elif config['net'] == 'resnet18':
            resnet = resnet18(weights = weights)
        else:
            raise NotImplementedError

        if config['data_name'] == 'poverty':
            resnet.conv1 = \
                nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3,bias=False)

        if 'feat_dim' in config.keys() and config['feat_dim'] != 0:
            # change feat_dim only if config[feat_dim]!=0
            # default feat_dim = 2048 (resnet50) / 512 (resnet34) / 512 (resnet18)
            if config['net'] == 'resnet50':
                resnet.layer4[2].conv3 = \
                    nn.Conv2d(512, config['feat_dim'], kernel_size=(1,1), stride=(1,1), bias=False)
                resnet.layer4[2].bn3 = \
                    nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                resnet.layer4[2].downsample = \
                    nn.Sequential(
                        nn.Conv2d(2048, config['feat_dim'], kernel_size=(1,1), stride=(1,1), bias=False),
                        nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                resnet.fc = nn.Linear(in_features=config['feat_dim'], out_features=2, bias=True)
            elif config['net'] == 'resnet34':
                resnet.layer4[2].conv2 = \
                    nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1), bias=False)
                resnet.layer4[2].bn2 = \
                    nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                resnet.layer4[2].downsample = \
                    nn.Sequential(
                        nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1),  bias=False),
                        nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                resnet.fc = nn.Linear(in_features=config['feat_dim'], out_features=2, bias=True)
            elif config['net'] == 'resnet18':
                resnet.layer4[1].conv2 = \
                    nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1), bias=False)
                resnet.layer4[1].bn2 = \
                    nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                resnet.layer4[1].downsample = \
                    nn.Sequential(
                        nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1),  bias=False),
                        nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                resnet.fc = nn.Linear(in_features=config['feat_dim'], out_features=2, bias=True)
            else:
                raise NotImplementedError

        fc_in_dim = resnet.fc.in_features
        resnet.fc = nn.Identity() # ignore fc layer
        super().__init__(config, resnet, fc_in_dim, nb_classes)


class ResNetRegress(Regress):
    def __init__(self, config):
        if 'pretrained' in config.keys() and config['pretrained'] == True:
            weights = 'DEFAULT'
        else:
            weights = None

        if config['net'] in ['resnet50_regress', 'resnet_regress']:
            resnet = resnet50(weights = weights)
        elif config['net'] == 'resnet34_regress':
            resnet = resnet34(weights = weights)
        elif config['net'] == 'resnet18_regress':
            resnet = resnet18(weights = weights)
        else:
            raise NotImplementedError

        # PovertyMap use 8 channel images
        if config['data_name'] == 'poverty':
            resnet.conv1 = \
                nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3,bias=False)

        if 'feat_dim' in config.keys() and config['feat_dim'] != 0 and not config['selective_train_regress']:
            # change feat_dim only if config[feat_dim]!=0
            # default feat_dim = 2048 (resnet50) / 512 (resnet34) / 512 (resnet18)
            if config['net'] in ['resnet50_regress', 'resnet_regress']:
                resnet.layer4[2].conv3 = \
                    nn.Conv2d(512, config['feat_dim'], kernel_size=(1,1), stride=(1,1), bias=False)
                resnet.layer4[2].bn3 = \
                    nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                resnet.layer4[2].downsample = \
                    nn.Sequential(
                        nn.Conv2d(2048, config['feat_dim'], kernel_size=(1,1), stride=(1,1), bias=False),
                        nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                resnet.fc = nn.Linear(in_features=config['feat_dim'], out_features=2, bias=True)
            elif config['net'] == 'resnet34_regress':
                resnet.layer4[2].conv2 = \
                    nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1), bias=False)
                resnet.layer4[2].bn2 = \
                    nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                resnet.layer4[2].downsample = \
                    nn.Sequential(
                        nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1),  bias=False),
                        nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                resnet.fc = nn.Linear(in_features=config['feat_dim'], out_features=2, bias=True)
            elif config['net'] == 'resnet18_regress':
                resnet.layer4[1].conv2 = \
                    nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1), bias=False)
                resnet.layer4[1].bn2 = \
                    nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                resnet.layer4[1].downsample = \
                    nn.Sequential(
                        nn.Conv2d(512, config['feat_dim'], kernel_size=(3,3), stride=(1,1),  bias=False),
                        nn.BatchNorm2d(config['feat_dim'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                resnet.fc = nn.Linear(in_features=config['feat_dim'], out_features=2, bias=True)
            else:
                raise NotImplementedError

        fc_in_dim = resnet.fc.in_features
        resnet.fc = nn.Identity() # ignore fc layer
        super().__init__(config, feature_extractor=resnet, fc_in_dim=fc_in_dim)


class TabResNetRegress(Component):
    """ Resnet for tabular data.
    """
    def __init__(self, config) -> None:
        super(TabResNetRegress, self).__init__(config, feature_extractor=None)

        #### hyperparameters obtained from "tuned" tabular revisting code ####
        d = 467
        d_hidden_factor = 2
        n_layers = 2 #config['res_layers']
        self.residual_dropout = 0.05
        self.hidden_dropout = 0.5
        ######################################################################

        d_in = config['d_in']
        normalization = 'batchnorm'
        activation = 'relu'
        d_out = 1

        def make_normalization(norm_d):
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](norm_d)

        self.main_activation = F.relu #lib.get_activation_fn(activation)
        self.last_activation = F.relu #lib.get_nonglu_activation_fn(activation)


        d_hidden = int(d * d_hidden_factor)

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(d),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.penultimate_normalization = make_normalization(d)
        if 'feat_dim' in config.keys():
            self.last_normalization = make_normalization(config['feat_dim'])
            self.feat = nn.Linear(d, config['feat_dim'])
            self.head = nn.Linear(config['feat_dim'], d_out)
        else:
            self.last_normalization = make_normalization(d)
            self.feat = nn.Linear(d, d)
            self.head = nn.Linear(d, d_out)
        self.to(self.device)

    def forward(self, x_num: Tensor, return_feature=False) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)

        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.penultimate_normalization(x)
        x = self.main_activation(x) #lib.get_activation_fn(activation)
        feat = self.feat(x)
        x = self.last_normalization(feat)
        x = self.last_activation(x)
        y_hat = self.head(x)
        if return_feature:
            return y_hat, feat
        return y_hat


class TabResNetClassify(Component):
    """ Resnet for tabular data.
    """
    def __init__(self,config, nb_classes=None) -> None:
        super(TabResNetClassify, self).__init__(config, feature_extractor=None)
        #### hyperparameters obtained from "tuned" tabular revisting code ####
        d = 256
        d_hidden_factor = 2
        n_layers = 2
        self.residual_dropout = 0.05
        self.hidden_dropout = 0.5
        ######################################################################

        d_in = config['d_in']
        normalization = 'batchnorm'
        activation = 'relu'

        if nb_classes == None:
            # categories: ty.Optional[ty.List[int]],
            categories = []
            for subset in config['data_schedule'][0]['subsets']:
                categories.append(subset[1])

            # if config['classification']:
                # d_embedding = config['d_embedding']
            d_out = len(categories)
        else:
            d_out = nb_classes

        def make_normalization(norm_d):
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](norm_d)

        self.main_activation = F.relu #lib.get_activation_fn(activation)
        self.last_activation = F.relu #lib.get_nonglu_activation_fn(activation)

        d_hidden = int(d * d_hidden_factor)

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(d),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.penultimate_normalization = make_normalization(d)
        if 'feat_dim' in config.keys():
            self.feat = nn.Linear(d, config['feat_dim'])
            self.last_normalization = make_normalization(config['feat_dim'])
            self.head = nn.Linear(config['feat_dim'], d_out)
        else:
            self.feat = nn.Linear(d, d)
            self.last_normalization = make_normalization(d)
            self.head = nn.Linear(d, d_out)

        self.to(self.device)

    def forward(self, x_num: Tensor, x_cat=None, return_feature=False) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.penultimate_normalization(x)
        x = self.main_activation(x) #lib.get_activation_fn(activation)
        feat = self.feat(x)
        x = self.last_normalization(feat)
        x = self.last_activation(x)
        y_hat = self.head(x)
        if return_feature:
            return y_hat, feat
        return y_hat

class TabResNetRegressFragment(Component):
    """ Resnet for tabular data.
    """
    def __init__(self, config) -> None:
        super(TabResNetRegressFragment, self).__init__(config, feature_extractor=None)

        #### hyperparameters obtained from "tuned" tabular revisting code ####
        d = 256
        d_hidden_factor = 2
        n_layers = 2 #config['res_layers']
        self.residual_dropout = 0.05
        self.hidden_dropout = 0.5
        ######################################################################

        d_in = config['d_in']
        normalization = 'batchnorm'
        activation = 'relu'
        d_out = 1

        def make_normalization(norm_d):
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](norm_d)

        self.main_activation = F.relu #lib.get_activation_fn(activation)
        self.last_activation = F.relu #lib.get_nonglu_activation_fn(activation)


        d_hidden = int(d * d_hidden_factor)

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(d),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.penultimate_normalization = make_normalization(d)
        if 'feat_dim' in config.keys():
            self.last_normalization = make_normalization(config['feat_dim'])
            self.feat = nn.Linear(d, config['feat_dim'])
            self.head = nn.Linear(config['feat_dim'], d_out)
        else:
            self.last_normalization = make_normalization(d)
            self.feat = nn.Linear(d, d)
            self.head = nn.Linear(d, d_out)
        self.to(self.device)

    def forward(self, x_num: Tensor, return_feature=False) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)

        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.penultimate_normalization(x)
        x = self.main_activation(x) #lib.get_activation_fn(activation)
        feat = self.feat(x)
        x = self.last_normalization(feat)
        x = self.last_activation(x)
        y_hat = self.head(x)
        if return_feature:
            return y_hat, feat
        return y_hat


class MLP(Regress):
    def __init__(self, config):
        """
        # modified from original architecture to handle shift15m
        mlp = nn.Sequential(nn.Flatten(),
                            nn.Linear(config['x_h'] * config['x_w'], config['h1_dim']),
                            nn.ReLU(),
                            nn.Linear(config['h1_dim'], config['h2_dim']),
                            nn.ReLU())
        """
        layers = [nn.Flatten()]
        prev_dim = config['feature_dim']
        for dim in config['fc_dim']:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        mlp = nn.Sequential(*layers)

        fc_in_dim = mlp[-2].out_features
        super().__init__(config, mlp, fc_in_dim)


class MLPClassify(Classify):
    def __init__(self, config, nb_classes):
        layers = [nn.Flatten()]
        prev_dim = config['feature_dim']
        for dim in config['fc_dim']:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        mlp = nn.Sequential(*layers)

        fc_in_dim = mlp[-2].out_features
        super().__init__(config, feature_extractor=mlp,
                         fc_in_dim=fc_in_dim, nb_classes = nb_classes)

