import sklearn.covariance
import torch
import scipy
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable

def random_sample_mean(feature, label, num_classes, frac=0.5):
    """
    returns mean, covariance of random samples
    this will be used as the initial value of MCD algorithm
    """

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered = False)
    total_sampled_feature, fraction_list = [], []
    sample_mean_per_class = torch.Tensor(num_classes, feature.size(1)).fill_(0).cuda()
    sample_precision_per_class = []
    label = label.cuda()

    # sampling features at each classes, and calculate its mean
    total_sampled_idcs_list = []
    for i in range(num_classes):
        cla_index = label.eq(i)
        cla_feature = feature[cla_index.nonzero(), :]
        cla_feature = cla_feature.view(cla_feature.size(0), -1)
        shuffler_idx = torch.randperm(cla_feature.size(0))
        sampled_index = shuffler_idx[:int(cla_feature.size(0)*frac)]
        fraction_list.append(int(cla_feature.size(0)*frac))
        total_sampled_idcs_list.append(cla_index.nonzero()[sampled_index.cuda()])

        sampled_feature = torch.index_select(cla_feature, 0, sampled_index.cuda())
        total_sampled_feature.append(sampled_feature)
        sample_mean_per_class[i].copy_(torch.mean(sampled_feature, 0))

    # calculate covariances at each class
    for i in range(num_classes):
        flag = 0
        X = 0
        for j in range(fraction_list[i]):
            sampled_feature = total_sampled_feature[i][j]
            sampled_feature = sampled_feature - sample_mean_per_class[i]
            sampled_feature = sampled_feature.view(-1,1)
            if flag  == 0:
                X = sampled_feature.transpose(0,1)
                flag = 1
            else:
                X = torch.cat((X,sampled_feature.transpose(0,1)),0)
            # find inverse
        group_lasso.fit(X.cpu().numpy())
        cov = group_lasso.covariance_

        # IMPORTANT NOTE: using float64 at scipy.linalg.pinvh returns wrong value
#        precision = scipy.linalg.pinvh(cov)
        precision = scipy.linalg.pinvh(cov.astype(np.float32))

        sample_precision_per_class.append(torch.from_numpy(precision).float().cuda())

    return sample_mean_per_class, sample_precision_per_class, total_sampled_idcs_list

def MCD_single(feature, mean, inverse_covariance, frac=0.5):
    """
    single run of MCD algorithm
    """
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    temp_batch = 100
    total, total_m_dist = 0, 0

    # calculate Mahalanobis distance for all features
    for data_index in range(int(np.ceil(feature.size(0)/temp_batch))):
        batch_feature = feature[total : total + temp_batch].cuda()
        zero_f = batch_feature - mean
        term_gau = -0.5*torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
        # concat data
        if total == 0:
            total_m_dist = term_gau.view(-1,1)
        else:
            total_m_dist = torch.cat((total_m_dist, term_gau.view(-1,1)), 0)
        total += batch_feature.size(0)

    total_m_dist = total_m_dist.view(-1)
    feature = feature.view(feature.size(0), -1)

    # select features with high Mahalanobis score
    topk_m_dist, selected_idx = torch.topk(total_m_dist, int(feature.size(0)*frac))
    selected_feature = torch.index_select(feature, 0, selected_idx.cuda())
    selected_feature_mean = torch.mean(selected_feature, 0)

    # compute covariance matrix
    X = 0
    flag = 0
    for j in range(selected_feature.size(0)):
        feature_j = selected_feature[j]
        feature_j = feature_j - selected_feature_mean
        feature_j = feature_j.view(-1,1)
        if flag  == 0:
            X = feature_j.transpose(0,1)
            flag = 1
        else:
            X = torch.cat((X, feature_j.transpose(0,1)),0)
    group_lasso.fit(X.cpu().numpy())
    cov = group_lasso.covariance_

    # find inverse
#    precision = scipy.linalg.pinvh(cov)
    precision = scipy.linalg.pinvh(cov.astype(np.float32))

    return selected_feature_mean, cov, selected_idx

@torch.no_grad()
def test_mcd_single(dist_mean, dist_precision, data, label_mask, gt_label_mask, selection_perc):
    """
    evaluate selection with given mean, precision
    """
    batch_size = 100
    total = 0
    total_num_data = data.size(0)
    m_dist = []

    # calculate Mahalanobis distance for all features
    for data_index in range(int(np.ceil(total_num_data/batch_size))):
        features = data[total : total + batch_size].cuda()
        features = Variable(features)

        zero_f = features - dist_mean
        term_gau = torch.mm(torch.mm(zero_f, dist_precision), zero_f.t()).diag()
        m_dist.append(term_gau.cpu().numpy())

        total += features.size(0)
    m_dist = np.concatenate(m_dist, axis=0)

    num_select = 0
    tp_clean = 0
    correct_lbl_perc_in_thld = []

    # get thld_val calcualted by selection_perc
    m_dist_lbl = m_dist[label_mask]
    thld_cnt = int(len(m_dist_lbl)*selection_perc)
    thld_val = np.partition(m_dist_lbl, thld_cnt)[thld_cnt]
    in_thld_mask = m_dist < thld_val

    # evaluation
    select_mask = np.logical_and(in_thld_mask, label_mask)
    num_select += np.sum(select_mask)
    clean_mask = label_mask == gt_label_mask
    tp_clean += np.sum(np.logical_and(select_mask, clean_mask))

    num_correct_lbl_in_thld = np.sum(np.logical_and(in_thld_mask, gt_label_mask))
    correct_lbl_perc_in_thld.append(num_correct_lbl_in_thld / np.sum(in_thld_mask))

    precision = tp_clean/num_select
    select_perc = num_select/np.sum(label_mask)

    return precision, select_perc, np.mean(correct_lbl_perc_in_thld)

@torch.no_grad()
def test_mcd(G_soft_layer, data, label, gt_label, selection_perc):
    """
    evaluate classification / selection with given G_soft_layer
    """
    batch_size = 100
    total, correct_D, correct_D_gt = 0, 0, 0
    total_num_data = data.size(0)
    total_m_dist = []

    # calculate Mahalanobis distances, num_correct for all features
    for data_index in range(int(np.ceil(total_num_data/batch_size))):
        features = data[total : total + batch_size].cuda()
        features = Variable(features)
        target = label[total : total + batch_size].cuda()
        target = Variable(target)
        target_gt = gt_label[total : total + batch_size].cuda()
        target_gt = Variable(target_gt)

        m_dist = G_soft_layer(features)
        likelihood = F.softmax(m_dist, dim=1)
        likelihood = Variable(likelihood.data)

        total += target.size(0)
        pred = likelihood.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        equal_flag_gt = pred.eq(target_gt.data).cpu()
        correct_D += equal_flag.sum()
        correct_D_gt += equal_flag_gt.sum()
        total_m_dist.append(m_dist.cpu().numpy())

    total_m_dist = np.concatenate(total_m_dist, axis=0)

    # evaluate classification
    acc = correct_D / total
    acc = acc.data
    acc_gt = correct_D_gt / total
    acc_gt = acc_gt.data

    clean_mask = (label == gt_label).numpy()

    # evaluate selection
    num_select = 0
    tp_clean = 0
    correct_lbl_perc_in_thld = []

    for lbl in [0, 1]:
        m_dist = total_m_dist[:,lbl]
        lbl_mask = label.numpy() == lbl

        m_dist_lbl = m_dist[lbl_mask]
        thld_cnt = int(len(m_dist_lbl)*selection_perc)
        thld_val = np.partition(m_dist_lbl, thld_cnt)[thld_cnt]
        in_thld_mask = m_dist < thld_val

        select_mask = np.logical_and(in_thld_mask, lbl_mask)
        num_select += np.sum(select_mask)
        tp_clean += np.sum(np.logical_and(select_mask, clean_mask))

        lbl_gt_mask = gt_label.numpy() == lbl
        num_correct_lbl_in_thld = np.sum(np.logical_and(in_thld_mask, lbl_gt_mask))
        correct_lbl_perc_in_thld.append(num_correct_lbl_in_thld / np.sum(in_thld_mask))

    precision = tp_clean/num_select
    select_perc = num_select/total_num_data

    return acc, acc_gt, precision, select_perc, np.mean(correct_lbl_perc_in_thld)

