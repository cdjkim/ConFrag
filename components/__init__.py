from typing import Dict
from .component import Component
from .network import ResNetRegress, ResNet, MLP, MLPClassify, \
    TabResNetRegress, TabResNetClassify, TabResNetRegressFragment

Net: Dict[str, Component] = {
    'mlp_regress': MLP,
    'mlp_classify': MLPClassify,
    'resnet50': ResNet,
    'resnet_classify': ResNet,
    'resnet18_regress': ResNetRegress,
    'resnet34_regress': ResNetRegress,
    'resnet50_regress': ResNetRegress,
    'resnet_regress': ResNetRegress,
    'tabresnet_regress': TabResNetRegress,
    'tabresnet_regress_frag': TabResNetRegressFragment,
    'tabresnet': TabResNetClassify,
}
