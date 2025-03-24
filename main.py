#!/usr/bin/env python3
from argparse import ArgumentParser
from pprint import pprint
import os
import resource
import random
import yaml
import torch
import colorful
import numpy as np
from tensorboardX import SummaryWriter
from data import DataScheduler, DataTaskScheduler
from models import MODEL
from train import train_model, filter_data, extract_feature
from utils import setup_logger, override_config
from timeit import default_timer as timer


# Increase maximum number of open files.
# as suggested in https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (24000, rlimit[1]))

parser = ArgumentParser()
parser.add_argument(
    '--random_seed', '-r', type=int, default=0)
parser.add_argument(
    '--config', '-c', default='configs/imdb_fragment.yaml'
)
parser.add_argument(
    '--episode', '-e', default='episodes/imdb-split4.yaml'
)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--resume-ckpt', default=None)
parser.add_argument('--override', default='')


def main():
    args = parser.parse_args()
    logger = setup_logger()

    # Use below for slurm setting.
    slurm_job_id = os.getenv('SLURM_JOB_ID', 'nojobid')
    slurm_proc_id = os.getenv('SLURM_PROC_ID', None)

    unique_identifier = str(slurm_job_id)
    if slurm_proc_id is not None:
        unique_identifier = unique_identifier + "_" + str(slurm_proc_id)

    # Load config
    config_path = args.config
    episode_path = args.episode

    if args.resume_ckpt and not args.config:
        base_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
        config_path = os.path.join(base_dir, 'config.yaml')
        episode_path = os.path.join(base_dir, 'episode.yaml')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    config = override_config(config, args.override)
    episode = yaml.load(open(episode_path), Loader=yaml.FullLoader)
    config['data_schedule'] = episode

    config['log_dir'] = os.path.join(args.log_dir, unique_identifier)

    # print the configuration
    print(colorful.bold_white("configuration:").styled_string)
    pprint(config)
    print(colorful.bold_white("configuration end").styled_string)

    if args.resume_ckpt and not args.log_dir:
        config['log_dir'] = os.path.dirname(
            os.path.dirname(args.resume_ckpt)
        )

    # set seed
    if args.random_seed != 0:
        config['random_seed'] = args.random_seed
    else:
        config['random_seed'] = random.randint(0, 1000)

    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    if not args.resume_ckpt or args.config:
        config_save_path = os.path.join(config['log_dir'], 'config.yaml')
        episode_save_path = os.path.join(config['log_dir'], 'episode.yaml')
        yaml.dump(config, open(config_save_path, 'w'))
        yaml.dump(episode, open(episode_save_path, 'w'))
        print(colorful.bold_yellow('config & episode saved to {}'.format(config['log_dir'])).styled_string)

    writer = SummaryWriter(config['log_dir'])

    # mem_filler = torch.rand((256,1024, 5000)).cuda() #XXX

    if 'fragment' in config['model_name']:
        # fragment model specific exp flow

        true_noise_rate = config['noise']['corrupt_p']
        data_scheduler = DataTaskScheduler(config, writer)
        config = data_scheduler.config

        model = MODEL[config['model_name']](config, data_scheduler, writer)
        if args.resume_ckpt:
            model.load_state_dict(torch.load(args.resume_ckpt))
        model.to(config['device'])

        if config['train_expert']:
            # training expert models
            print(colorful.bold_yellow(f'FRAGMENT: train experts').styled_string)
            train_model(config, model, data_scheduler, writer)

        if config['extract_feature']:
            # only extract feature from pretrained ckpt
            print(colorful.bold_yellow(f'FRAGMENT: extract features').styled_string)
            extract_feature(config, model, writer)

        if config['expert_filter_dataset']:
            # filtering dataset using trained experts or using experts' ckpt
            print(colorful.bold_yellow(f'FRAGMENT: filter dataset by experts').styled_string)
            assert config['filter_load'] == ''
            start = timer()
            filter_data(config, model, writer)
            end = timer()
            elapsed = end - start
            print(f"filter_data() time elapsed: {elapsed}")

        if config['selective_train_regress']:
            # training regressor based on selected samples
            print(colorful.bold_yellow(f'FRAGMENT: train regressor based on selected samples').styled_string)

            del model
            torch.cuda.empty_cache()
            # re-init random seed for noise reproducibility
            random.seed(config['random_seed'])
            np.random.seed(config['random_seed'])
            torch.manual_seed(config['random_seed'])

            config['classification'] = False
            config['model_name'] = config['regress_model']
            config['net'] = config['regress_net']
            config['loss'] = config['regress_loss']
            config['total_epochs'] = config['regress_total_epochs']
            config['batch_size'] = config['regress_batch_size']
            config['eval_every'] = 1
            config['optimizer'] = config['regress_optimizer']
            if config['net'] == 'mlp_regress':
                config['fc_dim'] = config['regress_fc_dim']
            config['noise']['corrupt_p'] = true_noise_rate

            # re-init DataTaskScheduler to train from single model with pred_clean_idcs.

            data_scheduler = DataScheduler(config, writer)
            config = data_scheduler.datasets[config['data_name']].config
            model = MODEL[config['model_name']](config, data_scheduler, writer)

            train_model(config, model, data_scheduler, writer)
    else:
        # general model exp flow

        data_scheduler = DataScheduler(config, writer)
        config = data_scheduler.datasets[config['data_name']].config

        model = MODEL[config['model_name']](config, data_scheduler, writer)
        if args.resume_ckpt:
            model.load_state_dict(torch.load(args.resume_ckpt))
        model.to(config['device'])

        train_model(config, model, data_scheduler, writer)


if __name__ == '__main__':
    main()
