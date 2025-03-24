import os
import torch
import colorful
import pickle as pkl
from tensorboardX import SummaryWriter
from data import DataScheduler
from timeit import default_timer as timer
from utils import get_exp_path



def train_model(config, model, scheduler: DataScheduler, writer: SummaryWriter):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')
    os.makedirs(saved_model_path, exist_ok=True)

    step = 0
    for (task_loader, epoch, t) in scheduler:
        step = model.learn(task_loader, epoch, t, step)

def extract_feature(config, model, writer: SummaryWriter):
    '''
    create /feature/ directory, and save extracted features, knns, likelihoods, total_idcs
    '''
    if config['expert_load'] not in ['', None]:
        # if exper_load is given, use the path
        saved_expert_ckpt_path = get_exp_path(config['expert_load'], 'ckpts', config['use_recent_identifier'])
    else:
        # if expert_load is not given, suppose it is saved at current log_dir
        saved_expert_ckpt_path = os.path.join(config['log_dir'], 'ckpts')
    print(colorful.bold_green(f"Load features from {saved_expert_ckpt_path}").styled_string)

    feature_save_path = os.path.join(config['log_dir'], 'feature')
    os.makedirs(feature_save_path, exist_ok=True)

    # check if all ckpts are available
    epc_list = list(range(config['save_every'], config['task_epochs'] + 1, config['save_every']))
    for epc in epc_list:
        for i in range(len(model.experts)):
            resume_ckpt = os.path.join(saved_expert_ckpt_path, f'ckpt-expert{i}-epc{epc}')
            if not os.path.exists(resume_ckpt):
                raise ValueError(f'classifier ckpt not valid {resume_ckpt}')

    # do extraction
    for epc in epc_list:
        print(colorful.bold_green(f'Extract features for epoch {epc}').styled_string)
        for i in range(len(model.experts)):
            resume_ckpt = os.path.join(saved_expert_ckpt_path, f'ckpt-expert{i}-epc{epc}')
            state_dict = torch.load(resume_ckpt)
            model.experts[i].load_state_dict(state_dict[f'expert_{i}'][0])
            model.experts[i].optimizer.load_state_dict(state_dict[f'optimizer_{i}'][0])
        model.to(config['device'])

        # get features, knns, likelihoods, total_idcs
        features, knns, likelihoods, y_hats, total_idcs, total_features = model.extract_feature()

        # save the returned values
        expert_feats_save_path = os.path.join(feature_save_path, f'features_{epc}.pkl')
        with open(expert_feats_save_path, 'wb') as handle:
            pkl.dump(features, handle, protocol=pkl.HIGHEST_PROTOCOL)

        expert_knns_save_path = os.path.join(feature_save_path, f'knns_{epc}.pkl')
        with open(expert_knns_save_path, 'wb') as handle:
            pkl.dump(knns, handle, protocol=pkl.HIGHEST_PROTOCOL)

        likelihood_save_path = os.path.join(feature_save_path, f'likelihood_{epc}.pkl')
        with open(likelihood_save_path, 'wb') as handle:
            pkl.dump(likelihoods, handle, protocol=pkl.HIGHEST_PROTOCOL)

        y_hat_save_path = os.path.join(feature_save_path, f'y_hat_{epc}.pkl')
        with open(y_hat_save_path, 'wb') as handle:
            pkl.dump(y_hats, handle, protocol=pkl.HIGHEST_PROTOCOL)

        idcs_save_path = os.path.join(feature_save_path, f'total_idcs_{epc}.pkl')
        with open(idcs_save_path, 'wb') as handle:
            pkl.dump(total_idcs, handle, protocol=pkl.HIGHEST_PROTOCOL)

        total_feats_save_path = os.path.join(feature_save_path, f'total_features_{epc}.pkl')
        with open(total_feats_save_path, 'wb') as handle:
            pkl.dump(total_features, handle, protocol=pkl.HIGHEST_PROTOCOL)

        if not config['save_features'] and config['train_expert']:
            # delete after filter for space maintenance.
            # individual delete for customizability later to leave best epoch.
            for i in range(len(model.experts)):
                resume_ckpt = os.path.join(saved_expert_ckpt_path, f'ckpt-expert{i}-epc{epc}')
                if os.path.exists(resume_ckpt):
                    os.remove(resume_ckpt)
                else:
                    print(f"The file {resume_ckpt} does not exist")

def filter_data(config, model, writer: SummaryWriter):
    if (config['expert_load'] not in ['', None]) and config['extract_feature'] == False:
        # if exper_load is given, use the path
        saved_feature_path = get_exp_path(config['expert_load'], 'feature',
                                          config['use_recent_identifier'])
    else:
        # if expert_load is not given, suppose it is saved at current log_dir
        saved_feature_path = os.path.join(config['log_dir'], 'feature')
    print(colorful.bold_green(f"Load features from {saved_feature_path}").styled_string)

    # filter results will be saved that filter_save_path
    filter_save_path = os.path.join(config['log_dir'], 'filter')
    os.makedirs(filter_save_path, exist_ok=True)

    # do filter
    epc_list = list(range(config['save_every'], config['task_epochs'] + 1,
                          config['save_every']))
    for epc in epc_list:
        print(colorful.bold_green(f'Filter for epoch {epc}').styled_string)

        # load features, knns, likelihoods, total_ids
        features = {}
        for instance_name in ['features', 'knns', 'likelihood', 'total_idcs',
                              'y_hat', 'total_features']:
            try:
                instance_saved_path = os.path.join(saved_feature_path,
                                                   f'{instance_name}_{epc}.pkl')
                with open(instance_saved_path, 'rb') as handle:
                    features[instance_name] = pkl.load(handle)
            except Exception as e:
                print(e)

        pred_as_clean_idcs = model.filter(features, epc)

        clean_idcs_save_path = os.path.join(filter_save_path, f'clean_idcs_{epc}.pkl')
        with open(clean_idcs_save_path, 'wb') as handle:
            pkl.dump(pred_as_clean_idcs, handle, protocol=pkl.HIGHEST_PROTOCOL)

        if not config['save_weights'] and (config['train_expert']
                                           or config['extract_feature']):
            # delete after filter for space maintenance.
            for instance_name in ['features', 'knns', 'likelihood', 'total_idcs',
                                  'y_hat', 'total_features']:
                instance_saved_path = os.path.join(saved_feature_path,
                                                   f'{instance_name}_{epc}.pkl')
                if os.path.exists(instance_saved_path):
                    os.remove(instance_saved_path)
                else:
                    print(f"The file {instance_saved_path} does not exist")

