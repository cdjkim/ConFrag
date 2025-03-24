from copy import deepcopy
import numpy as np
import random
import os
import re
import pickle as pkl
import colorful
import hashlib
import typing as ty
import sklearn
from collections import Counter, defaultdict
from sklearn.impute import SimpleImputer

ArrayDict = ty.Dict[str, np.ndarray]

def new_inject_noise(targets, noise_config):
    """
    This new inject_noise method is upgraded in two aspects
    1. It can handle 'multi-label'
    2. It can handle 'continuous value'
        * previous method deals with targets as categorical value(Inject noise in agedb/imdb)
    """
    noise_type = noise_config['type']
    corrupt_p = noise_config['corrupt_p']
    std = noise_config['std']

    noisy_targets = deepcopy(targets)
    num_noise_sample = int(corrupt_p * targets.shape[0])

    if noise_type == 'symmetric':
        # symmetric or no noise.
        sorted_idx = np.arange(targets.shape[0])
        shuffled_idx = np.random.permutation(sorted_idx)
        idx = shuffled_idx[:round(len(shuffled_idx) * corrupt_p)]

        while True:
            noisy_targets[idx] = targets[np.random.choice(targets.shape[0], len(idx))]
            noisy_mask = (np.sum(targets[idx]==noisy_targets[idx], axis=1) != targets.shape[1])
            if np.sum(~noisy_mask) == 0:
                break
            idx = idx[~noisy_mask]

        noise_rate = np.sum(np.any(noisy_targets != targets, axis=1)) / len(targets)
        print(colorful.bold_red(f"Symmetric noise rate: {noise_rate}").styled_string)
        noise_error = np.mean(abs(noisy_targets - targets))
        print(colorful.bold_red(f"Gaussian random noise error: {noise_error:.4f}"))

    elif noise_type == 'gaussian_random':
        # bin-based gaussian random noise (for efficient injection)
        # gaussian noise at bin level -> random sampling from the bin
        num_bin = 4000
        min_t = np.min(targets)
        max_t = np.max(targets)
        total_range = max_t - min_t

        def _get_bin_idx(n):
            if n == max_t:
                return num_bin - 1
            else:
                return int((n - min_t) / total_range * num_bin)

        noisy_idx = []
        unique_targets = np.unique(targets)

        bin_idx_to_uniq_t = defaultdict(list)
        for t in unique_targets:
            bin_idx_to_uniq_t[_get_bin_idx(t)].append(t)

        for t in unique_targets:
            std = np.random.randint(low = noise_config['std_min'], high = noise_config['std_max'])

            cls_idx = np.where(np.array(targets) == t)[0]
            num_noise_sample = int(corrupt_p * cls_idx.shape[0])
            sampled_cls_idx = np.random.choice(cls_idx, num_noise_sample, replace=False)

            for idx in sampled_cls_idx:
                noisy_targets[idx] = np.random.normal(t, scale=std)
                sample_try = 0
                while True:
                    sampled_t_raw = np.random.normal(t, scale=std)
                    if sampled_t_raw >= min_t and sampled_t_raw <= max_t:
                        sampled_bin_idx = _get_bin_idx(sampled_t_raw)
                        if len(bin_idx_to_uniq_t[sampled_bin_idx]) != 0:
                            break

                    sample_try += 1
                    if sample_try > 40:
                        sampled_t_raw = t
                        sampled_bin_idx = _get_bin_idx(sampled_t_raw)
                        break

                sampled_t = np.random.choice(bin_idx_to_uniq_t[sampled_bin_idx])

                noisy_targets[idx] = sampled_t

        # print resulting noise percentage.
        noise_rate_original = np.sum(noisy_targets != targets) / len(targets)
        noise_rate = np.sum(abs(noisy_targets - targets) >= (targets.max()-targets.min())/8) / len(targets)
        noise_error = np.mean(abs(noisy_targets - targets))
        print(colorful.bold_red(f"Gaussian random noise rate: {noise_rate_original:.4f} / soft noise rate: {noise_rate:.4f}").styled_string)
        print(colorful.bold_red(f"Gaussian random noise error: {noise_error:.4f}"))
    else:
        raise NotImplementedError

    return noisy_targets, noise_rate

def inject_noise(targets, noise_config):
    noise_type = noise_config['type']
    corrupt_p = noise_config['corrupt_p']
    std = noise_config['std']

    noisy_targets = deepcopy(targets)

    unique_targets = np.unique(targets)

    if noise_type == 'symmetric':
        # symmetric or no noise.
        for t in unique_targets:
            random_target = list(unique_targets)
            random_target.remove(t)
            t_idx = [i for i, y in enumerate(targets) if y == t]
            random.shuffle(t_idx)
            for i in t_idx[:round(len(t_idx)*corrupt_p)]:
                noisy_targets[i] = random.choice(random_target)

        # print resulting noise percentage.
        noise_rate = np.sum(noisy_targets != targets) / len(targets)
        print(colorful.bold_red(f"Symmetric noise rate: {noise_rate}").styled_string)

    elif noise_type == 'gaussian_random':
        noisy_idx = []

        if isinstance(unique_targets[0], np.int32) or isinstance(unique_targets[0], int) \
                or isinstance(unique_targets[0], np.int64):
            for t in unique_targets:
                std = np.random.randint(low = noise_config['std_min'], high = noise_config['std_max'])

                cls_idx = np.where(np.array(targets) == t)[0]
                num_noise_sample = int(corrupt_p * cls_idx.shape[0])
                sampled_cls_idx = np.random.choice(cls_idx, num_noise_sample, replace=False)
                for idx in sampled_cls_idx:
                    sample_try = 0
                    while True:
                        sampled_t = np.round(np.random.normal(t, scale=std))
                        if corrupt_p == 1.0:
                            if sampled_t in unique_targets:
                                break
                        else:
                            if sampled_t in unique_targets and sampled_t != t:
                                break

                        sample_try += 1
                        # if sampling failed for 40 times, just use original target
                        if sample_try > 40:
                            sampled_t = t

                    noisy_targets[idx] = sampled_t
        elif isinstance(unique_targets[0], np.floating) or isinstance(unique_targets[0], float):
            # when dealing with MSD or decimal unique targets, with equidistant label space.
            # make unique_targets the idcs.
            unique_targets_idcs = list(range(0, len(unique_targets)))
            for t, t_idx in zip(unique_targets, unique_targets_idcs):
                std = np.random.randint(low = noise_config['std_min'], high = noise_config['std_max'])
                cls_idx = np.where(np.array(targets) == t)[0]
                num_noise_sample = int(corrupt_p * cls_idx.shape[0])
                sampled_cls_idx = np.random.choice(cls_idx, num_noise_sample, replace=False)
                for idx in sampled_cls_idx:
                    sample_try = 0
                    while True:
                        sampled_t = int(np.round(np.random.normal(t_idx, scale=std)))
                        if corrupt_p == 1.0:
                            if sampled_t in unique_targets_idcs:
                                break
                        else:
                            if sampled_t in unique_targets_idcs and sampled_t != t_idx: #?? second condition why?
                                break

                        sample_try += 1
                        # if sampling failed for 40 times, just use original target
                        if sample_try > 40:
                            sampled_t = t_idx

                    noisy_targets[idx] = unique_targets[sampled_t]

        # print resulting noise percentage.
        noise_rate_original = np.sum(noisy_targets != targets) / len(targets)
        noise_rate = np.sum(abs(noisy_targets - targets) >= (targets.max()-targets.min())/8) / len(targets)
        noise_error = np.mean(abs(noisy_targets - targets))
        print(colorful.bold_red(f"Gaussian random noise rate: {noise_rate_original:.4f} / soft noise rate: {noise_rate:.4f}").styled_string)
        print(colorful.bold_red(f"Gaussian random noise error: {noise_error:.4f}"))
    else:
        raise NotImplementedError

    return noisy_targets, noise_rate

def unique_target_fragmentation(noisy_targets, org_targets, label_split, split,
                                label_coverage=1.0):
    """Fragments the continuous unique targets into discrete targets
    Args:
        noisy_targets(np.array)
        org_targets(np.array)
        label_split(list): [num_split]
        split(str): train/val/test
        label_coverage(float): percentage of the label space to cover.
    Returns:
        clf_targets(list of list)
        gt_clf_targets(list of list)
        split_map(dict)
    """
    # for handling both 1-d and N-d label
    if len(org_targets.shape) == 1:
        org_targets = np.expand_dims(org_targets, axis=1)
        noisy_targets = np.expand_dims(noisy_targets, axis=1)
    clf_targets = []
    gt_clf_targets = []
    split_map = {}
    total_regr_noise_num = 0
    total_clf_noise_num = 0
    dim_clf_y_start = 0

    for dim_i, l_split in enumerate(label_split):
        split_clf_targets = []
        split_gt_clf_targets = []
        unq = np.unique(noisy_targets[:, dim_i])

        # 1. decide on label_coverage. buffer + label_coverage = total label range
        cover_num = int(len(unq) * label_coverage)
        cover_unq = unq[:cover_num]
        buffer = unq[cover_num:]
        # 2. randomly decide how much to shift within the buffer.
        if len(buffer) > 0:
            shift = random.randint(0, len(buffer))
        else:
            shift = 0
        # 3. set the final shifted label coverage
        final_unq = unq[shift:shift + cover_num]
        # split_map style modified:
        # {DIM_i: {clf_0:range_0, clf_1:range_1, clf_2:range_2, clf_3:range_3},
        #  DIM_j: {clf_4:range_4, clf_5:range_5, clf_6:range_6, clf_7:range_7}}
        split_map[dim_i] = {dim_clf_y_start + split_i: split_range
                            for split_i, split_range in enumerate(
                            np.array_split(final_unq, l_split))}

        clf_y_lst = np.array(list(split_map[dim_i].keys()))
        for lbl, gt_lbl in zip(noisy_targets[:, dim_i], org_targets[:, dim_i]):
            if lbl not in final_unq: #or gt_lbl not in final_unq:
                continue
            for clf_y, split_range in split_map[dim_i].items():
                if lbl in split_range:
                    split_clf_targets.append(clf_y)
                    cur_clf_y = clf_y
                if gt_lbl in split_range:
                    split_gt_clf_targets.append(clf_y)
                    cur_clf_y = clf_y
                if gt_lbl not in split_range and lbl in split_range:
                    total_clf_noise_num += 1
            # when gt_lbl is in excluded jittered range, but lbl is in it
            # assign anything else than clf_y assigned to lbl since this is noise anyways.
            if gt_lbl not in final_unq and label_coverage < 1.0: # and lbl is
                split_gt_clf_targets.append(np.delete(clf_y_lst, cur_clf_y)[0])
            if gt_lbl != lbl:
                total_regr_noise_num += 1

        clf_targets.append(split_clf_targets)
        gt_clf_targets.append(split_gt_clf_targets)
        dim_clf_y_start += l_split

    assert len(clf_targets[dim_i]) == len(gt_clf_targets[dim_i])

    print(colorful.bold_red(f"Regr noise rate: "\
                            f"{total_regr_noise_num/len(org_targets)}\
                            split {split}").styled_string)
    print(colorful.bold_red(f"Post Clf conversion noise rate: "\
                            f"{total_clf_noise_num/len(clf_targets[0])}\
                            split {split}").styled_string)

    return clf_targets, gt_clf_targets, split_map

def range_target_fragmentation(noisy_targets, org_targets, label_split, split,
                                label_coverage=1.0):
    """Fragments the continuous target ranges into discrete targets
       for handling both 1-d and N-d label.
    Args:
        noisy_targets(np.array)
        org_targets(np.array)
        label_split(list): [num_split]
        split(str): train/val/test
        label_coverage(float): percentage of the label space to cover.
    Returns:
        clf_targets(list of list)
        gt_clf_targets(list of list)
        split_map(dict)
    """
    if len(org_targets.shape) == 1:
        org_targets = np.expand_dims(org_targets, axis=1)
        noisy_targets = np.expand_dims(noisy_targets, axis=1)
    clf_targets = []
    gt_clf_targets = []
    split_map = {}
    total_regr_noise_num = 0
    total_clf_noise_num = 0
    dim_clf_y_start = 0

    for dim_i, l_split in enumerate(label_split):
        split_clf_targets = []
        split_gt_clf_targets = []
#        split_map[dim_i] = np.array_split(unq, l_split)
        # split_map style modified:
        # {DIM_i: {clf_0:range_0, clf_1:range_1, clf_2:range_2, clf_3:range_3},
        #  DIM_j: {clf_4:range_4, clf_5:range_5, clf_6:range_6, clf_7:range_7}}
        t_sort = np.sort(org_targets[:,dim_i])
        len_t = len(t_sort)

        # 1. decide on label_coverage.
        cover_num = int(len_t * label_coverage)
        # 2. randomly decide how much to shift within the buffer.
        buffer = t_sort[cover_num:]
        if len(buffer) > 0:
            shift = random.randint(0, len(buffer) - 1)
        else:
            shift = 0
        # 3. set the final shifted label coverage
        t_sort = t_sort[shift:shift + cover_num]
        len_t = len(t_sort)

        t_sort = np.append(t_sort, t_sort[-1]+1) # +1 buffer.

        final_range = ()
        # NOTE below idx made using int, makes the actual covered num different from cover_num.
        split_map[dim_i] = {}
        for s in range(l_split):
            split_range = (t_sort[int(s/l_split*len_t)], t_sort[int((s+1)/l_split*len_t)])
            split_map[dim_i][dim_clf_y_start + s] = split_range

        final_range = (t_sort[int(0/l_split*len_t)], t_sort[int(l_split/l_split*len_t)])

        clf_y_lst = np.array(list(split_map[dim_i].keys()))
        for lbl, gt_lbl in zip(noisy_targets[:, dim_i], org_targets[:, dim_i]):
            if lbl < final_range[0] or lbl >= final_range[1]:
                # skip label not in final range
                continue
            for clf_y, split_range in split_map[dim_i].items():
                if lbl >= split_range[0] and lbl < split_range[1]:
                    split_clf_targets.append(clf_y)
                    cur_clf_y = clf_y
                if gt_lbl >= split_range[0] and gt_lbl < split_range[1]:
                    split_gt_clf_targets.append(clf_y)
                    cur_clf_y = clf_y
                elif lbl >= split_range[0] and lbl < split_range[1]:
                    total_clf_noise_num += 1
            if label_coverage < 1.0:
                if gt_lbl < final_range[0] or gt_lbl >= final_range[1]:
                    split_gt_clf_targets.append(np.delete(clf_y_lst, cur_clf_y)[0])
            if gt_lbl != lbl:
                total_regr_noise_num += 1

        clf_targets.append(split_clf_targets)
        gt_clf_targets.append(split_gt_clf_targets)
        dim_clf_y_start += l_split

    assert len(clf_targets[dim_i]) == len(gt_clf_targets[dim_i])

    print(colorful.bold_red(f"Regr noise rate: "\
                            f"{total_regr_noise_num/len(org_targets)} \
                            split {split}").styled_string)
    print(colorful.bold_red(f"Post Clf conversion noise rate: "\
                            f"{total_clf_noise_num/len(clf_targets[0])} \
                            split {split}").styled_string)

    return clf_targets, gt_clf_targets, split_map


def filter_dataset(saved_dir):
    max_epc = 0
    for file in os.listdir(os.path.join(saved_dir, 'pred')):
        if file.startswith('clean_idcs_'):
            num = int(re.search('clean_idcs_(\d*)', file).group(1))
            max_epc = num if num > max_epc else max_epc
    clean_idcs_save_path = os.path.join(saved_dir, f"pred/clean_idcs_{max_epc}.pkl")
    print(colorful.bold_green(f"reading pred_clean_dics_{max_epc}.pkl file...").styled_string)
    with open(clean_idcs_save_path, 'rb') as handle:
        clean_idcs = pkl.load(handle)
    return clean_idcs

def save_noise_checksum(save_dir, noisy_data, overwrite=True):
    # to check if noisy data is equal without saving the whole array
    os.makedirs(save_dir, exist_ok=True)

    noisy_data_checksum = hashlib.sha1(str(noisy_data).encode("utf-8")).hexdigest()

    checksum_file_path = os.path.join(save_dir, 'noise_checksum.txt')
    if os.path.exists(checksum_file_path) and overwrite==False:
        raise ValueError(f"Checksum file already exists: {checksum_file_path}")

    with open(checksum_file_path, 'w') as f:
        f.write(noisy_data_checksum)

def verify_noise_checksum(save_dir, targets):
    # check if checksum is correct
    checksum_file_path = os.path.join(save_dir, 'noise_checksum.txt')
    if not os.path.exists(checksum_file_path):
        print(colorful.bold_red(f"Noise checksum file does not exist"))
        return False

    with open(checksum_file_path, 'r') as f:
        saved_checksum = f.readline()

    current_checksum = hashlib.sha1(str(targets).encode("utf-8")).hexdigest()
    return current_checksum == saved_checksum

def build_X(
    N,
    C,
    data_path,
    normalization: ty.Optional[str],
    num_nan_policy: str,
    cat_nan_policy: str,
    cat_policy: str,
    cat_min_frequency: float = 0.0,
    seed: int = 2,
) -> ty.Union[ArrayDict, ty.Tuple[ArrayDict, ArrayDict]]:
    cache_path = (
        data_path
        / f'build_X__{normalization}__{num_nan_policy}__{cat_nan_policy}__{cat_policy}__{seed}.pickle'  # noqa
        if data_path
        else None
    )
    if cache_path and cat_min_frequency:
        cache_path = cache_path.with_name(
            cache_path.name.replace('.pickle', f'__{cat_min_frequency}.pickle')
        )
    if cache_path and cache_path.exists():
        print(f'Using cached X: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pkl.load(f)

    def save_result(x):
        if cache_path:
            with open(cache_path, 'wb') as f:
                pkl.dump(x, f)

    if N:
        N = deepcopy(N)

        num_nan_masks = {k: np.isnan(v) for k, v in N.items()}
        if any(x.any() for x in num_nan_masks.values()):  # type: ignore[code]
            if num_nan_policy == 'mean':
                num_new_values = np.nanmean(N['train'], axis=0)
            else:
                raise ValueError(f'numerical NaN policy {num_nan_policy} unknown')
            for k, v in N.items():
                num_nan_indices = np.where(num_nan_masks[k])
                v[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])
        if normalization:
            N = normalize(N, normalization, seed)

    else:
        N = None

    if cat_policy == 'drop' or not C:
        assert N is not None
        save_result(N)
        return N

    C = deepcopy(C)

    cat_nan_masks = {k: v == 'nan' for k, v in C.items()}
    if any(x.any() for x in cat_nan_masks.values()):  # type: ignore[code]
        if cat_nan_policy == 'new':
            cat_new_value = '___null___'
            imputer = None
        elif cat_nan_policy == 'most_frequent':
            cat_new_value = None
            imputer = SimpleImputer(strategy=cat_nan_policy)  # type: ignore[code]
            imputer.fit(C['train'])
        else:
            raise ValueError(f'categorical NaN policy {cat_nan_policy} unknown')

        if imputer:
            C = {k: imputer.transform(v) for k, v in C.items()}
        else:
            for k, v in C.items():
                cat_nan_indices = np.where(cat_nan_masks[k])
                v[cat_nan_indices] = cat_new_value

    if cat_min_frequency:
        C = ty.cast(ArrayDict, C)
        min_count = round(len(C['train']) * cat_min_frequency)
        rare_value = '___rare___'
        C_new = {x: [] for x in C}
        for column_idx in range(C['train'].shape[1]):
            counter = Counter(C['train'][:, column_idx].tolist())
            popular_categories = {k for k, v in counter.items() if v >= min_count}
            for part in C_new:
                C_new[part].append(
                    [
                        (x if x in popular_categories else rare_value)
                        for x in C[part][:, column_idx].tolist()
                    ]
                )
        C = {k: np.array(v).T for k, v in C_new.items()}

    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(C['train'])
    C = {k: encoder.transform(v) for k, v in C.items()}
    max_values = C['train'].max(axis=0)
    for part in ['val', 'test']:
        for column_idx in range(C[part].shape[1]):
            C[part][C[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    if cat_policy == 'indices':
        result = (N, C)
    elif cat_policy == 'ohe':
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype='float32'  # type: ignore[code]
        )
        ohe.fit(C['train'])
        C = {k: ohe.transform(v) for k, v in C.items()}
        result = C if N is None else {x: np.hstack((N[x], C[x])) for x in N}
    else:
        raise ValueError(f'categoriccl policy {cat_policy} is unknown')
    save_result(result)
    return result  # type: ignore[code]

def build_y(
    y_data, policy: ty.Optional[str]
) -> ty.Tuple[ArrayDict, ty.Optional[ty.Dict[str, ty.Any]]]:
    y = deepcopy(y_data)
    if policy:
        if policy == 'mean_std':
            mean, std = y_data['train'].mean(), y_data['train'].std()
            y = {k: (v - mean) / std for k, v in y.items()}
            info = {'policy': policy, 'mean': mean, 'std': std}
        else:
            raise ValueError(f'y policy {policy} is unknown')
    else:
        info = None
    return y, info

def normalize(
    X: ArrayDict, normalization: str, seed: int, noise: float = 1e-3
) -> ArrayDict:
    X_train = X['train'].copy()
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            # subsample=1e9,
            subsample=1_000_000_000,
            random_state=seed,
        )
        if noise:
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train += noise_std * np.random.default_rng(seed).standard_normal(  # type: ignore[code]
                X_train.shape
            )
    else:
        raise ValueError(f'normalization {normalization} is unknown')
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}  # type: ignore[code]
