import os
from copy import deepcopy
import tqdm
import time
import torch
import colorful
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from components import Net
from utils import CRITERION, write_grad_norm_scalar, mixup, KNN, count_parameters
from more_itertools import chunked
import faiss
from scipy.special import softmax

import random


class Fragment(torch.nn.Module):
    def __init__(self, config, scheduler, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.writer = writer
        self.scheduler = scheduler
        if self.config['loss'] in ['mse', 'l1']:
            self.disc_type = 'regressor'
        else:
            self.disc_type = 'classifier'

        self.criterion = CRITERION[config['loss']](config, writer)
        self.experts, self.experts_subset_info = self.get_init_experts(config)
        self.noise_rate = self.config['noise']['corrupt_p']
        self.saved_model_path = os.path.join(config['log_dir'], 'ckpts')
        self.dataset = self.scheduler.datasets[self.config['data_name']]

        self.ingred = {}

    def get_init_experts(self, config):
        """get initialized expert models"""
        experts = []
        schedule = self.scheduler.schedule
        experts_subset_info = []
        for i in range(len(schedule)):
            if self.disc_type == 'regressor':
                expert = Net[config['net']](config)
            elif config['net'] in ['resnet50', 'resnet34', 'resnet18']:
                nb_classes = len(schedule[i]['subsets'])
                expert = Net["resnet_classify"](config, nb_classes=nb_classes)
            else:
                nb_classes = len(schedule[i]['subsets'])
                expert = Net[config['net']](config, nb_classes=nb_classes)
            optim_config = config['optimizer']
            lr_scheduler_config = deepcopy(config['lr_scheduler'])
            if lr_scheduler_config['type'] == 'CosineAnnealingLR':
                lr_scheduler_config['options'].update({'T_max': config['task_epochs']})

            expert.setup_optimizer(optim_config)
            expert.setup_lr_scheduler(lr_scheduler_config)

            expert.train()
            print(colorful.bold_blue(f'# of model params : {count_parameters(expert)} / original: 617,474'))
            experts.append(expert)
            experts_subset_info.append(schedule[i]['subsets'])

        print(colorful.bold_green(f'num of experts {len(experts)}').styled_string)

        return experts, experts_subset_info

    def get_eval_model(self, t):
        expert = self.experts[t]
        expert.eval()
        return expert

    def get_eval_dataset(self, t):
        test_dataset = self.scheduler.test_task_datasets[t]
        clf_y_dim = self.scheduler.task_i_to_dim[t]
        return test_dataset, clf_y_dim

    def learn(self, task_loader, epoch, t, step):
        expert = self.experts[t]
        task_info = self.scheduler.schedule[t]['subsets']
        clf_y_dim = self.scheduler.task_i_to_dim[t]

        for epc_i in tqdm.trange(epoch, desc='epoch', leave=False):
            if self.disc_type == 'regressor':
                self.epoch_clean_abs_err = 0
                self.epoch_noisy_abs_err = 0
            else:
                self.epoch_clean_correct = 0
                self.epoch_noisy_correct = 0
            self.epoch_clean_total = 0
            self.epoch_noisy_total = 0
            total = 0
            if self.config['jitter']:
                task_loader = self.scheduler.datasets[self.config['data_name']]\
                    .jitter_data(t, task_info, self.config['label_coverage'])

            for x, clf_y, gt_clf_y, y, gt_y, idx in tqdm.tqdm(task_loader,
                                                              desc='step',
                                                              leave=True):
                x = x.to(self.device)
                y = torch.unsqueeze(y, dim=1)
                expert.zero_grad(set_to_none=True)
                clf_y_train = clf_y[clf_y_dim].clone()
                gt_clf_y_train = gt_clf_y[clf_y_dim]
                clean_mask = (clf_y_train == gt_clf_y_train).to(self.device)
                for new_lbl, (_, clf_lbl) in enumerate(task_info):
                    clf_y_train[clf_y[clf_y_dim]==clf_lbl] = new_lbl

                clf_y_train = clf_y_train.to(self.device)

                # vanilla
                if len(x) == 1:
                    # if the number of sample is 1, error occurs at batch_norm
                    continue
                y_hat = expert(x)

                if self.disc_type == 'regressor':
                    loss = self.compute_regressor_loss(y_hat, y, gt_y, step, epc_i)
                else:
                    loss = self.compute_loss(y_hat, clf_y_train, clean_mask, y,
                                             gt_y, idx, step, epc_i, expert_i=t)

                mean_loss = torch.mean(loss)
                mean_loss.backward()

                if self.config['tensorboard']['grad']:
                    write_grad_norm_scalar(self.writer, expert, step)

                expert.optimizer.step()
                self.writer.add_scalar(f'train/loss', mean_loss, step)

                total += x.shape[0]
                step += 1

            expert.lr_scheduler.step()
            self.criterion.flush(epc_i, postfix=f'_{t}')

            if self.config['verbose']:
                if self.disc_type == 'regressor':
                    clean_train_mae = self.epoch_clean_abs_err / self.epoch_clean_total
                    self.writer.add_scalar(f'train/clean_mae/expert_{t}',
                                           clean_train_mae, epc_i)
                    if self.config['noise']['corrupt_p'] > 0.0:
                        noisy_train_mae = self.epoch_noisy_abs_err / self.epoch_noisy_total
                        self.writer.add_scalar(f'train/noisy_mae/expert_{t}',
                                               noisy_train_mae, epc_i)
                else:
                    clean_train_accuracy = self.epoch_clean_correct / self.epoch_clean_total
                    self.writer.add_scalar(f'train/clean_accuracy/expert_{t}',
                                           clean_train_accuracy, epc_i)
                    if self.config['noise']['corrupt_p'] > 0.0:
                        noisy_train_accuracy = self.epoch_noisy_correct / self.epoch_noisy_total
                        self.writer.add_scalar(f'train/noisy_accuracy/expert_{t}',
                                               noisy_train_accuracy, epc_i)

            # Can skip eval for train speed.
            if (epc_i+1)%self.config['eval_every'] == 0 or epc_i+1 == epoch:

                self.scheduler.eval(self, self.writer, epc_i, t, task_info)

            # must save in order to eval later.
            if (epc_i+1)%self.config['save_every'] == 0 or epc_i+1 == epoch:
                state_dict = {}
                state_dict[f'expert_{t}'] =  expert.state_dict(),
                state_dict[f'optimizer_{t}'] = expert.optimizer.state_dict(),

                name_format = str(epc_i + 1)
                torch.save(state_dict,
                        os.path.join(self.saved_model_path,
                                        f'ckpt-expert{t}-epc{name_format}'))

        # return all indices to full data coverage.
        if self.config['jitter']:
            self.scheduler.datasets[self.config['data_name']]\
                .update_fragmentation(label_coverage=1.0, split='train')
        return step

    def compute_loss(self, y_hat, clf_y, clean_mask, y, gt_y, idx, step, epc_i, expert_i):
        clf_y = clf_y.squeeze()

        loss = self.criterion(input=y_hat, target=clf_y, reduction='none',
                                clean_mask=clean_mask, idx=idx, epoch=epc_i, step=step)
        pred = y_hat.argmax(dim=1)
        if self.config['noise']['corrupt_p'] > 0.0 and self.config['verbose']:
            # print wrong label loss and clean label loss separately.
            num_clean = torch.sum(clean_mask)
            clean_loss = torch.sum(loss * clean_mask) / num_clean
            self.writer.add_scalar(f'train/clean_loss', clean_loss, step)
            num_noisy = clean_mask.size(0) - num_clean
            if num_noisy > 0:
                noisy_loss = torch.sum(loss * ~clean_mask) / num_noisy
                self.writer.add_scalar(f'train/noisy_loss', noisy_loss, step)
            correct_mask = pred == clf_y
            clean_correct = (clean_mask * correct_mask).sum()
            noise_correct = (~clean_mask * correct_mask).sum()
        else:
            clean_idcs = gt_y
            noisy_idcs = []
            noise_correct = 0
            clean_correct = (pred == clf_y).float().sum()
            num_clean = clean_mask.size(0)
            num_noisy = 0

        if self.config['verbose']:
            self.epoch_clean_correct += clean_correct
            self.epoch_noisy_correct += noise_correct
            self.epoch_clean_total += num_clean
            self.epoch_noisy_total += num_noisy

        return loss

    def compute_regressor_loss(self, y_hat, y, gt_y, step, epc_i):
        gt_y = gt_y.squeeze()
        y = y.squeeze()
        clean_mask = (y == gt_y).to(self.device)

        y_hat = y_hat.to(self.device)
        y = y.to(self.device)

        target = y
        loss = self.criterion(input=y_hat, target=target, reduction='none')

        num_clean = torch.sum(clean_mask)
        clean_loss = torch.sum(loss.squeeze() * clean_mask) / num_clean if num_clean != 0 else 0
        self.writer.add_scalar(f'train/clean_loss', clean_loss, step)
        num_noisy = clean_mask.size(0) - num_clean
        noisy_loss = torch.sum(loss.squeeze() * ~clean_mask) / num_noisy if num_noisy != 0 else 0
        self.writer.add_scalar(f'train/noisy_loss', noisy_loss, step)
        abs_err = torch.abs(y_hat.squeeze() - target)
        clean_abs_err = (clean_mask * abs_err).sum()
        noise_abs_err = (~clean_mask * abs_err).sum()

        if self.config['verbose']:
            self.epoch_clean_abs_err += clean_abs_err
            self.epoch_noisy_abs_err += noise_abs_err
            self.epoch_clean_total += num_clean
            self.epoch_noisy_total += num_noisy

        return loss

    @torch.no_grad()
    def extract_feature(self):
        """
        extract features, knns, likelihoods, total_idcs
        """
        expert_knns = {}
        expert_feats = {}
        likelihoods_save = []
        y_hat_save = []
        idx_save = []
        total_feats = []

        self.dataset.set_transform('eval')
        for expert in self.experts:
            expert.eval()

        for t, (_, task) in enumerate(self.scheduler.task_datasets):
            print(colorful.green('Progress to Task %d' % t).styled_string)
            clf_y_dim = self.scheduler.task_i_to_dim[t]
            collate_fn = task.datasets[0].dataset.collate_fn
            task_loader = DataLoader(task, batch_size=self.config['batch_size'],
                            num_workers=self.config['num_workers'],
                            collate_fn=collate_fn,
                            drop_last=False,
                            pin_memory=True, # better when training on GPU.
                            shuffle=False)

            expert_knns[t] = KNN(k=self.config['knn_k'])
            cur_t_feats = []
            cur_t_clf_y = []
            cur_t_gt_clf_y = []
            cur_t_gt_y = []
            cur_t_y = []
            cur_t_idx = []
            for x, clf_y, gt_clf_y, y, gt_y, idx in tqdm.tqdm(task_loader, desc='feat_extract&knn_fit', leave=False):
                x = x.to(self.device)
                likelihoods_t = []
                y_hat_t = []
                feats = []
                for t_i, expert in enumerate(self.experts):
                    y_hat, feat = expert(x, return_feature=True)
                    if t_i == t:
                        cur_t_feat = feat
                    feats.append(feat.cpu().numpy())
                    likelihoods_t.append(torch.nn.functional.softmax(y_hat, dim=1).cpu().numpy())
                    y_hat_t.append(y_hat.cpu().numpy())

                total_feats.append(np.concatenate(feats, axis=1))
                cur_t_feats.append(cur_t_feat.cpu().numpy())
                cur_t_clf_y.extend(clf_y[clf_y_dim].numpy())
                cur_t_gt_clf_y.extend(gt_clf_y[clf_y_dim].numpy())
                cur_t_idx.extend(idx.numpy())
                cur_t_gt_y.extend(gt_y.numpy())
                cur_t_y.extend(y.numpy())

                if self.disc_type == 'classifier':
                    likelihoods_save.append(np.concatenate(likelihoods_t, axis=1))
                y_hat_save.append(np.concatenate(y_hat_t, axis=1))

            idx_save.extend(cur_t_idx)
            # construct KNN graph on this task.
            expert_feats[t] = np.concatenate(cur_t_feats, axis=0)
            assert len(expert_feats[t]) == len(cur_t_clf_y) == len(cur_t_gt_clf_y) == len(cur_t_y) == len(cur_t_gt_y) == len(cur_t_idx)
            expert_knns[t].fit(expert_feats[t], cur_t_clf_y, cur_t_gt_clf_y, cur_t_y,
                               cur_t_gt_y, cur_t_idx, dist_metric="cosine",
                               normalize_feature=True)

        total_feats = np.concatenate(total_feats, axis=0)
        if self.disc_type == 'classifier':
            likelihoods_save = np.concatenate(likelihoods_save, axis=0)
        else:
            likelihoods_save = []
        y_hat_save = np.concatenate(y_hat_save, axis=0)
        idx_save = np.array(idx_save)

        return expert_feats, expert_knns, likelihoods_save, y_hat_save, idx_save, total_feats

    def filter(self, features, epc):
        """
        operate feature/prediction space filter
        """
        if self.config['frag_method'] == 'union_eta':
            filter_idcs = self.filter_union_eta(features, epc)
        elif self.config['frag_method'] == 'union_eta_regr':
            filter_idcs = self.filter_union_eta_regr(features, epc)
        else:
            raise NotImplementedError

        return filter_idcs

    def pred_space_filter(self, features, epc):
        '''
        filter dataset at prediction space
        '''
        print(f'[prediction space filter]')

        dataset = self.dataset
        num_classes = self.config['label_split'][0]

        total_idcs = features['total_idcs']
        y_hat = features['y_hat']
        assert y_hat.shape[1]%2 == 0
        likelihoods = []
        for c in range(int(y_hat.shape[1]//2)):
            likelihoods.append(softmax(y_hat[:,2*c:2*c+2], axis=1))
        likelihoods = np.concatenate(likelihoods, axis=1)

        lls = np.zeros((len(total_idcs), num_classes))

        # consider only single dimensional output
        assert len(self.config['label_split']) == 1

        # get orig_label_seq which will be used to rearange likelihood order
        orig_label_seq = []
        for task_i, subset_info in enumerate(self.experts_subset_info):
            orig_label_seq.extend([subset_info[0][1], subset_info[1][1]])
        assert len(orig_label_seq) == self.config['label_split'][0]

        for i, idx in enumerate(total_idcs):
            ll = [0] * len(orig_label_seq)
            for lik_idx, ll_idx in enumerate(orig_label_seq):
                ll[ll_idx] = likelihoods[i, lik_idx]
            lls[idx] = ll

        self.ingred['lls'] = lls

        return lls

    def feat_space_filter(self, features, epc):
        print(f'[feature space filter]')


        k_thld = self.config['k_threshold']
        feat_dim = self.config['feat_dim']

        expert_knns = features['knns']
        total_feats = features['total_features']
        total_idcs = features['total_idcs']
        num_classes = self.config['label_split'][0]
        num_task = len(self.scheduler.task_datasets)

        feat_dim = int(total_feats.shape[1] / num_task)

        task_to_clf_target = {}
        for task_i, subset_info in enumerate(self.experts_subset_info):
            task_to_clf_target[task_i] = (subset_info[1][1], subset_info[0][1])

        # enable this part to run knn on GPU
        if 'faiss_gpu' in self.config.keys() and self.config['faiss_gpu']:
            for t in expert_knns:
                expert_knns[t].index_cpu_to_gpu()

        # for each feature: query each of the KNN graphs to filter/discard.
        knn_prob = np.zeros((len(self.dataset), num_classes))
        knn_count = np.zeros((len(self.dataset), num_classes))

        n_idcs_total, n_dist_total = {}, {}
        n_gt_y_total, n_y_total = {}, {}
        n_gt_clf_y_total, n_clf_y_total = {}, {}

        for t in range(num_task):
            t_feats = total_feats[:,t*feat_dim:(t+1)*feat_dim]
            feature_idcs =range(t_feats.shape[0])

            for batch_i in tqdm.tqdm(chunked(feature_idcs, 512), desc=f'task{t}: knn filter', leave=False):

                query_node = t_feats[batch_i]
                faiss.normalize_L2(query_node)
                n_dist, _, n_meta = expert_knns[t].query(query_node, batch_i,
                                                         query_in_neighbor=False,
                                                         k=max(self.config['knn_k'], self.config['k_threshold'])+1)
                n_y, n_gt_y, n_clf_y, n_gt_clf_y, n_idcs = n_meta

                for i in range(len(batch_i)):
                    idx = total_idcs[batch_i[i]]


                    if n_idcs[i,0] == idx:
                        # when the 1st nearest neighbor is the query sample
                        n_y_i = n_y[i,1:]
                        n_gt_y_i = n_gt_y[i,1:]
                        n_clf_y_i = n_clf_y[i,1:]
                        n_gt_clf_y_i = n_gt_clf_y[i,1:]
                        n_idcs_i = n_idcs[i,1:]
                        n_dist_i = n_dist[i,1:]
                    else:
                        n_y_i = n_y[i]
                        n_gt_y_i = n_gt_y[i]
                        n_clf_y_i = n_clf_y[i]
                        n_gt_clf_y_i = n_gt_clf_y[i]
                        n_idcs_i = n_idcs[i]
                        n_dist_i = n_dist[i]

                    n_idcs_total[idx] = n_idcs_i
                    n_dist_total[idx] = n_dist_i
                    n_gt_y_total[idx] = n_gt_y_i
                    n_y_total[idx] = n_y_i
                    n_clf_y_total[idx] = n_clf_y_i
                    n_gt_clf_y_total[idx] = n_gt_clf_y_i

                    for clf_t in task_to_clf_target[t]:
                        num_agreed = np.sum(n_clf_y_i[:k_thld] == clf_t)
                        knn_prob[idx, clf_t] = np.exp(1 / k_thld * (num_agreed))
                        knn_count[idx, clf_t] = num_agreed

        self.ingred['n_idcs_total'] = n_idcs_total
        self.ingred['n_dist_total'] = n_dist_total
        self.ingred['n_gt_y_total'] = n_gt_y_total
        self.ingred['n_y_total'] = n_y_total
        self.ingred['n_gt_clf_y_total'] = n_gt_clf_y_total
        self.ingred['n_clf_y_total'] = n_clf_y_total

        avg_knn_prob = np.mean(knn_prob)
        avg_knn_count = np.mean(knn_count)
        print(f'avg_knn_prob {avg_knn_prob:.3f} / avg_knn_count {avg_knn_count:.3f}')
        self.writer.add_scalar(f'details/knn/avg_knn_prob', avg_knn_prob, epc)
        self.writer.add_scalar(f'details/knn/avg_knn_count', avg_knn_count, epc)

        return knn_prob, knn_count

    def filter_union_eta(self, features, epc):
        dataset = self.dataset
        clf_targets = np.array(dataset.clf_targets[0])
        gt_clf_targets = np.array(dataset.gt_clf_targets[0])
        targets = dataset.targets

        # get clean_masks
        clf_clean_mask = clf_targets == gt_clf_targets
        clean_mask = dataset.targets == dataset.org_targets

        # generate task to clf_target mapping
        task_to_clf_target = {}
        clf_target_to_task = {}
        paired_clf_target = {}
        for task_i, subset_info in enumerate(self.experts_subset_info):
            task_to_clf_target[task_i] = (subset_info[1][1], subset_info[0][1])
            clf_target_to_task[subset_info[0][1]] = task_i
            clf_target_to_task[subset_info[1][1]] = task_i
            paired_clf_target[subset_info[0][1]] = subset_info[1][1]
            paired_clf_target[subset_info[1][1]] = subset_info[0][1]

        # collect feat/pred space filter results
        knn_prob, knn_count = self.feat_space_filter(features, epc)
        lls = self.pred_space_filter(features, epc)

        org_split_map = self.dataset.split_map[0]
        data_range = np.max(targets) - np.min(targets)

        def _mean_split(lbl):
            return (max(org_split_map[lbl]) + min(org_split_map[lbl]))/2

        filter_cnt = 0
        tp_clf_clean = 0
        tp_clean = 0
        num_pred_only, num_knn_only, num_intersect = 0, 0, 0
        tp_clf_clean_pred_only, tp_clf_clean_knn_only, tp_clf_clean_intersect = 0, 0, 0

        filter_idcs = []
        filter_err = []

        for idx in range(len(self.dataset)):
            ll = lls[idx]
            knn_cnt = knn_count[idx]
            clf_y = clf_targets[idx]
            y = targets[idx]

            eps=0.1
            closeness = []
            random.shuffle([0,1])
            for j in range(len(ll)):
                closeness.append(data_range/(abs(y - _mean_split(j))+eps))
            eta_list = softmax(closeness)

            # pred sampling
            p_pred = 0
            for j in range(len(ll)):
                j_ngb_list = [j-1, j+1]
                random.shuffle(j_ngb_list)

                # get self agreement
                alpha_self = 1 if ll[j] >= ll[paired_clf_target[j]] else 0
                alpha_ngb_list = []
                for j_ngb in j_ngb_list:
                    if j_ngb < 0 or j_ngb >= len(ll):
                        # return False for out of boundary
                        alpha_ngb_list.append(0)
                    else:
                        alpha_ngb = 1 if ll[j_ngb] >= ll[paired_clf_target[j_ngb]] else 0
                        alpha_ngb_list.append(alpha_ngb)
                alpha_ngb_union = alpha_ngb_list[0] | alpha_ngb_list[1]

                p_pred += eta_list[j] * alpha_self * alpha_ngb_union

            # probability based sampling
            if random.random() <= p_pred:
                do_select_pred = True
            else:
                do_select_pred = False

            # knn sampling
            p_knn = 0
            for j in range(len(knn_cnt)):
                k_thld = self.config['k_threshold']
                j_ngb_list = [j-1, j+1]
                random.shuffle(j_ngb_list)

                # get self agreement
                alpha_self = 1 if knn_cnt[j] in [k_thld, k_thld-1] else 0

                # get neighbor agreement
                alpha_ngb_list = []
                for j_ngb in j_ngb_list:
                    if j_ngb < 0 or j_ngb >= len(knn_cnt):
                        # return False for out of boundary
                        alpha_ngb_list.append(0)
                    else:
                        alpha_ngb = 1 if knn_cnt[j_ngb] in [k_thld, k_thld-1] else 0
                        alpha_ngb_list.append(alpha_ngb)
                alpha_ngb_union = alpha_ngb_list[0] | alpha_ngb_list[1]

                p_knn += eta_list[j] * alpha_self * alpha_ngb_union

            # probability based sampling
            if random.random() <= p_knn:
                do_select_knn = True
            else:
                do_select_knn = False


            # combining rule: union
            do_select = do_select_pred | do_select_knn

            if do_select:
                filter_cnt += 1
                filter_idcs.append(idx)

                if clf_clean_mask[idx]:
                    tp_clf_clean += 1
                if clean_mask[idx]:
                    tp_clean += 1

                filter_err.append(abs(dataset.targets[idx] - dataset.org_targets[idx]))

                if do_select_pred and not do_select_knn:
                    num_pred_only += 1
                    if clf_clean_mask[idx]:
                        tp_clf_clean_pred_only += 1

                elif not do_select_pred and do_select_knn:
                    num_knn_only += 1
                    if clf_clean_mask[idx]:
                        tp_clf_clean_knn_only += 1

                elif do_select_pred and do_select_knn:
                    num_intersect += 1
                    if clf_clean_mask[idx]:
                        tp_clf_clean_intersect += 1

        filter_percentage = filter_cnt/len(self.dataset)
        clf_precision = tp_clf_clean/filter_cnt
        precision = tp_clean/filter_cnt
        filter_err = np.mean(filter_err)

        print(f'filter_perc {filter_percentage:.3f} / filter_err {filter_err:.3f} / clf_precision {clf_precision:.3f} / precision {precision:.3f}')

        self.writer.add_scalar(f'fragment/filter/filter_percentage', filter_percentage, epc)
        self.writer.add_scalar(f'fragment/filter/filter_err', filter_err, epc)
        self.writer.add_scalar(f'fragment/filter/clf_precision', clf_precision, epc)
        self.writer.add_scalar(f'fragment/filter/precision', precision, epc)

        pred_only_percentage = num_pred_only / len(self.dataset)
        pred_only_precision = tp_clf_clean_pred_only / num_pred_only if num_pred_only != 0 else 0
        knn_only_percentage = num_knn_only / len(self.dataset)
        knn_only_precision = tp_clf_clean_knn_only / num_knn_only if num_knn_only != 0 else 0
        intersect_percentage = num_intersect / len(self.dataset)
        intersect_precision = tp_clf_clean_intersect / num_intersect if num_intersect != 0 else 0
        self.writer.add_scalar(f'details/filter/pred_only_percentage', pred_only_percentage, epc)
        self.writer.add_scalar(f'details/filter/pred_only_precision', pred_only_precision, epc)
        self.writer.add_scalar(f'details/filter/knn_only_percentage', knn_only_percentage, epc)
        self.writer.add_scalar(f'details/filter/knn_only_precision', knn_only_precision, epc)
        self.writer.add_scalar(f'details/filter/intersect_percentage', intersect_percentage, epc)
        self.writer.add_scalar(f'details/filter/intersect_precision', intersect_precision, epc)

        return filter_idcs

    def filter_union_eta_regr(self, features, epc):
        """
        union_sgl, union_dbl
        """
        dataset = self.dataset
        clf_targets = np.array(dataset.clf_targets[0])
        gt_clf_targets = np.array(dataset.gt_clf_targets[0])
        targets = dataset.targets

        num_split = self.config['label_split'][0]

        # get clean_masks
        clf_clean_mask = clf_targets == gt_clf_targets
        clean_mask = dataset.targets == dataset.org_targets

        # generate task to clf_target mapping
        task_to_clf_target = {}
        clf_target_to_task = {}
        paired_clf_target = {}
        for task_i, subset_info in enumerate(self.experts_subset_info):
            task_to_clf_target[task_i] = (subset_info[0][1], subset_info[1][1])
            clf_target_to_task[subset_info[0][1]] = task_i
            clf_target_to_task[subset_info[1][1]] = task_i
            paired_clf_target[subset_info[0][1]] = subset_info[1][1]
            paired_clf_target[subset_info[1][1]] = subset_info[0][1]

        # collect feat/pred space filter results
        knn_prob, knn_count = self.feat_space_filter(features, epc)

        # pred_space_filter for regr is not required. only y_hat is enough
        total_idcs = features['total_idcs']
        y_hat = features['y_hat']
        preds = np.zeros((len(total_idcs), 2))
        for i, idx in enumerate(total_idcs):
            preds[idx] = y_hat[i]

        org_split_map = self.dataset.split_map[0]
        data_range = np.max(targets) - np.min(targets)

        def _get_clf_pred(pred, t):
            # clf_pred is decieded by closer fragment
            clf_target_0 = task_to_clf_target[t][0]
            clf_target_1 = task_to_clf_target[t][1]
            mean_clf_target_0 = np.mean(org_split_map[clf_target_0])
            mean_clf_target_1 = np.mean(org_split_map[clf_target_1])
            mean_task = (mean_clf_target_0 + mean_clf_target_1)/2

            if pred[t] >= mean_task:
                clf_pred = clf_target_1
            else:
                clf_pred = clf_target_0
            return clf_pred

        def _pred_agree(pred, clf_target):
            if clf_target not in clf_target_to_task.keys():
                # clf_target out of range returns false (not agreed)
                return False

            cur_task = clf_target_to_task[clf_target]
            if _get_clf_pred(pred, cur_task) == clf_target:
                return True
            else:
                return False

        def _get_neighbor(lbl, likelihoods, coverage=1, debug=False):
            # get neighbor lbl (choosen based on max(lik))
            max_lik = -1
            max_lik_lbl = []
            for i_lbl, i_lik in enumerate(likelihoods):
                if i_lbl == lbl:
                    # pass for self check
                    continue

                if abs(lbl - i_lbl) > coverage:
                    # out of coverage
                    continue

                if i_lik > max_lik:
                    max_lik = i_lik
                    max_lik_lbl = [i_lbl]
                elif i_lik == max_lik:
                    # this will not happen normally
                    # but might be happening at knn_cnt based approach
                    max_lik_lbl.append(i_lbl)

            assert len(max_lik_lbl) != 0, f'unexpected lbl {lbl} / likelihoods {likelihoods}'

            return random.choice(max_lik_lbl)

        def _mean_split(lbl):
            return (max(org_split_map[lbl]) + min(org_split_map[lbl]))/2

        filter_cnt = 0
        tp_clf_clean = 0
        tp_clean = 0
        num_pred_only, num_knn_only, num_intersect = 0, 0, 0
        tp_clf_clean_pred_only, tp_clf_clean_knn_only, tp_clf_clean_intersect = 0, 0, 0

        filter_idcs = []
        filter_err = []

        for idx in range(len(self.dataset)):
            pred = preds[idx]
            knn_cnt = knn_count[idx]
            y = targets[idx]

            # distance to y_tilde_c based approach
            # number of valid eta == F
            eps=0.1
            closeness = []
            for j in range(num_split):
                closeness.append(data_range/(abs(y - _mean_split(j))+eps))
            eta_list = softmax(closeness)

            # pred sampling
            p_pred = 0
            for j in range(num_split):
                alpha_self = _pred_agree(pred, j)
                alpha_ngb = _pred_agree(pred, j-1) | _pred_agree(pred, j+1)
                p_pred += eta_list[j] * alpha_self * alpha_ngb

            # probability based sampling
            if random.random() <= p_pred:
                do_select_pred = True
            else:
                do_select_pred = False



            # knn sampling
            p_knn = 0
            for j in range(len(knn_cnt)):
                k_thld = self.config['k_threshold']
                j_ngb_list = [j-1, j+1]
                random.shuffle(j_ngb_list)

                # get self agreement
                alpha_self = 1 if knn_cnt[j] in [k_thld, k_thld-1] else 0

                # get neighbor agreement
                alpha_ngb_list = []
                for j_ngb in j_ngb_list:
                    if j_ngb < 0 or j_ngb >= len(knn_cnt):
                        # return False for out of boundary
                        alpha_ngb_list.append(0)
                    else:
                        alpha_ngb = 1 if knn_cnt[j_ngb] in [k_thld, k_thld-1] else 0
                        alpha_ngb_list.append(alpha_ngb)
                alpha_ngb_union = alpha_ngb_list[0] | alpha_ngb_list[1]

                p_knn += eta_list[j] * alpha_self * alpha_ngb_union

            # probability based sampling
            if random.random() <= p_knn:
                do_select_knn = True
            else:
                do_select_knn = False

            # combining rule: union
            do_select = do_select_pred | do_select_knn

            if do_select:
                filter_cnt += 1
                filter_idcs.append(idx)

                if clf_clean_mask[idx]:
                    tp_clf_clean += 1
                if clean_mask[idx]:
                    tp_clean += 1

                filter_err.append(abs(dataset.targets[idx] - dataset.org_targets[idx]))

                if do_select_pred and not do_select_knn:
                    num_pred_only += 1
                    if clf_clean_mask[idx]:
                        tp_clf_clean_pred_only += 1

                elif not do_select_pred and do_select_knn:
                    num_knn_only += 1
                    if clf_clean_mask[idx]:
                        tp_clf_clean_knn_only += 1

                elif do_select_pred and do_select_knn:
                    num_intersect += 1
                    if clf_clean_mask[idx]:
                        tp_clf_clean_intersect += 1

        filter_percentage = filter_cnt/len(self.dataset)
        clf_precision = tp_clf_clean/filter_cnt
        precision = tp_clean/filter_cnt
        filter_err = np.mean(filter_err)

        print(f'filter_perc {filter_percentage:.3f} / filter_err {filter_err:.3f} / clf_precision {clf_precision:.3f} / precision {precision:.3f}')

        self.writer.add_scalar(f'fragment/filter/filter_percentage', filter_percentage, epc)
        self.writer.add_scalar(f'fragment/filter/filter_err', filter_err, epc)
        self.writer.add_scalar(f'fragment/filter/clf_precision', clf_precision, epc)
        self.writer.add_scalar(f'fragment/filter/precision', precision, epc)

        pred_only_percentage = num_pred_only / len(self.dataset)
        pred_only_precision = tp_clf_clean_pred_only / num_pred_only if num_pred_only != 0 else 0
        knn_only_percentage = num_knn_only / len(self.dataset)
        knn_only_precision = tp_clf_clean_knn_only / num_knn_only if num_knn_only != 0 else 0
        intersect_percentage = num_intersect / len(self.dataset)
        intersect_precision = tp_clf_clean_intersect / num_intersect if num_intersect != 0 else 0
        self.writer.add_scalar(f'details/filter/pred_only_percentage', pred_only_percentage, epc)
        self.writer.add_scalar(f'details/filter/pred_only_precision', pred_only_precision, epc)
        self.writer.add_scalar(f'details/filter/knn_only_percentage', knn_only_percentage, epc)
        self.writer.add_scalar(f'details/filter/knn_only_precision', knn_only_precision, epc)
        self.writer.add_scalar(f'details/filter/intersect_percentage', intersect_percentage, epc)
        self.writer.add_scalar(f'details/filter/intersect_precision', intersect_precision, epc)

        return filter_idcs

