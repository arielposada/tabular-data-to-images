"""Core IGTD (Image Generator for Tabular Data) algorithm.

Adapted from the official reference implementation at
https://github.com/zhuyitan/IGTD (MIT License), accompanying:

    Zhu, Y., Brettin, T., Xia, F., Partin, A., Shukla, M., Yoo, H., Evrard, Y. A.,
    Doroshow, J. H., & Stevens, R. L. (2021). Converting tabular data into images
    for deep learning with convolutional neural networks. Scientific Reports, 11,
    11325. https://doi.org/10.1038/s41598-021-90923-y

Trimmed to the functions this project actually calls (feature selection by
variance and the squared-error search variant were dropped), and adapted:
  - `IGTD_absolute_error`'s `random_state` argument now actually seeds NumPy's
    global RNG (the upstream code constructs a `np.random.RandomState` object
    without ever using it, so the requested seed was silently ignored).
  - `save_folder`/`image_folder` are optional everywhere (upstream unconditionally
    called `os.path.exists(None)`, which raises); passing `None` now just skips
    writing the optional diagnostic files.
  - `np.int` (removed in NumPy >= 1.24) replaced with `int`.
"""

import os
import time as _time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform


def min_max_transform(data):
    """Linearly rescale each feature (column) of `data` to the [0, 1] range."""
    norm_data = np.empty(data.shape)
    norm_data.fill(np.nan)
    for i in range(data.shape[1]):
        v = data[:, i].copy()
        if np.max(v) == np.min(v):
            norm_data[:, i] = 0
        else:
            v = (v - np.min(v)) / (np.max(v) - np.min(v))
            norm_data[:, i] = v
    return norm_data


def generate_feature_distance_ranking(data, method="Pearson"):
    """Rank pairwise feature dissimilarities (n_features x n_features matrix)."""
    num = data.shape[1]
    if method == "Pearson":
        corr = np.corrcoef(np.transpose(data))
        # Zero-variance (constant) columns make corrcoef undefined (NaN) for every
        # pair involving them; treat those as uncorrelated with everything rather
        # than letting NaN propagate through the whole ranking/optimization.
        corr = np.nan_to_num(corr, nan=0.0)
    elif method == "Euclidean":
        corr = squareform(pdist(np.transpose(data), metric="euclidean"))
        corr = np.max(corr) - corr
        corr = corr / np.max(corr)
    else:
        raise ValueError(f"Unsupported method: {method}")

    corr = 1 - corr
    corr = np.around(a=corr, decimals=10)

    tril_id = np.tril_indices(num, k=-1)
    rank = rankdata(corr[tril_id])
    ranking = np.zeros((num, num))
    ranking[tril_id] = rank
    ranking = ranking + np.transpose(ranking)

    return ranking, corr


def generate_matrix_distance_ranking(num_r, num_c, method="Euclidean", num=None):
    """Rank pairwise pixel distances for a num_r x num_c image grid."""
    if num is None:
        num = num_r * num_c

    for r in range(num_r):
        if r == 0:
            coordinate = np.transpose(np.vstack((np.zeros(num_c), range(num_c))))
        else:
            coordinate = np.vstack(
                (coordinate, np.transpose(np.vstack((np.ones(num_c) * r, range(num_c)))))
            )
    coordinate = coordinate[:num, :]

    cord_dist = np.zeros((num, num))
    if method == "Euclidean":
        for i in range(num):
            cord_dist[i, :] = np.sqrt(
                np.square(coordinate[i, 0] * np.ones(num) - coordinate[:, 0])
                + np.square(coordinate[i, 1] * np.ones(num) - coordinate[:, 1])
            )
    elif method == "Manhattan":
        for i in range(num):
            cord_dist[i, :] = np.abs(
                coordinate[i, 0] * np.ones(num) - coordinate[:, 0]
            ) + np.abs(coordinate[i, 1] * np.ones(num) - coordinate[:, 1])
    else:
        raise ValueError(f"Unsupported method: {method}")

    tril_id = np.tril_indices(num, k=-1)
    rank = rankdata(cord_dist[tril_id])
    ranking = np.zeros((num, num))
    ranking[tril_id] = rank
    ranking = ranking + np.transpose(ranking)

    coordinate = np.int64(coordinate)
    return (coordinate[:, 0], coordinate[:, 1]), ranking


def IGTD_absolute_error(
    source,
    target,
    max_step=1000,
    switch_t=0,
    val_step=50,
    min_gain=0.00001,
    random_state=1,
    save_folder=None,
    file_name="",
    verbose=True,
):
    """Greedy local search: reorder `source`'s rows/columns to match `target`.

    Returns (index_record, err_record, run_time) — see the module docstring's
    upstream reference for full details. `index_record[-1]` (or, more robustly,
    `index_record[np.argmin(err_record)]`) is the best feature ordering found.
    """
    np.random.seed(random_state)
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    source = source.copy()
    num = source.shape[0]
    tril_id = np.tril_indices(num, k=-1)
    index = np.array(range(num))
    index_record = np.empty((max_step + 1, num))
    index_record.fill(np.nan)
    index_record[0, :] = index.copy()

    err_v = np.empty(num)
    err_v.fill(np.nan)
    for i in range(num):
        err_v[i] = np.sum(np.abs(source[i, 0:i] - target[i, 0:i])) + np.sum(
            np.abs(source[(i + 1):, i] - target[(i + 1):, i])
        )

    step_record = -np.ones(num)
    err_record = [np.sum(abs(source[tril_id] - target[tril_id]))]
    pre_err = err_record[0]
    t1 = _time.time()
    run_time = [0]

    for s in range(max_step):
        delta = -np.ones(num) * np.inf
        idr = np.where(step_record == np.min(step_record))[0]
        ii = idr[np.random.permutation(len(idr))[0]]

        for jj in range(num):
            if jj == ii:
                continue
            if ii < jj:
                i, j = ii, jj
            else:
                i, j = jj, ii

            err_ori = err_v[i] + err_v[j] - np.abs(source[j, i] - target[j, i])
            err_i = (
                np.sum(np.abs(source[j, :i] - target[i, :i]))
                + np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i]))
                + np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i]))
                + np.abs(source[i, j] - target[j, i])
            )
            err_j = (
                np.sum(np.abs(source[i, :i] - target[j, :i]))
                + np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j]))
                + np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j]))
                + np.abs(source[i, j] - target[j, i])
            )
            err_test = err_i + err_j - np.abs(source[i, j] - target[j, i])
            delta[jj] = err_ori - err_test

        delta_norm = delta / pre_err
        candidates = np.where(delta_norm >= switch_t)[0]
        if len(candidates) > 0:
            jj = np.argmax(delta)
            if ii < jj:
                i, j = ii, jj
            else:
                i, j = jj, ii

            for k in range(num):
                if k < i:
                    err_v[k] = (
                        err_v[k]
                        - np.abs(source[i, k] - target[i, k])
                        - np.abs(source[j, k] - target[j, k])
                        + np.abs(source[j, k] - target[i, k])
                        + np.abs(source[i, k] - target[j, k])
                    )
                elif k == i:
                    err_v[k] = (
                        np.sum(np.abs(source[j, :i] - target[i, :i]))
                        + np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i]))
                        + np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i]))
                        + np.abs(source[i, j] - target[j, i])
                    )
                elif k < j:
                    err_v[k] = (
                        err_v[k]
                        - np.abs(source[k, i] - target[k, i])
                        - np.abs(source[j, k] - target[j, k])
                        + np.abs(source[k, j] - target[k, i])
                        + np.abs(source[i, k] - target[j, k])
                    )
                elif k == j:
                    err_v[k] = (
                        np.sum(np.abs(source[i, :i] - target[j, :i]))
                        + np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j]))
                        + np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j]))
                        + np.abs(source[i, j] - target[j, i])
                    )
                else:
                    err_v[k] = (
                        err_v[k]
                        - np.abs(source[k, i] - target[k, i])
                        - np.abs(source[k, j] - target[k, j])
                        + np.abs(source[k, j] - target[k, i])
                        + np.abs(source[k, i] - target[k, j])
                    )

            ii_v = source[ii, :].copy()
            jj_v = source[jj, :].copy()
            source[ii, :] = jj_v
            source[jj, :] = ii_v
            ii_v = source[:, ii].copy()
            jj_v = source[:, jj].copy()
            source[:, ii] = jj_v
            source[:, jj] = ii_v
            err = pre_err - delta[jj]

            t = index[ii]
            index[ii] = index[jj]
            index[jj] = t

            step_record[ii] = s
            step_record[jj] = s
        else:
            err = pre_err
            step_record[ii] = s

        err_record.append(err)
        if verbose and (s % 50 == 0):
            print(f"IGTD step {s}: err={err:.4f}")
        index_record[s + 1, :] = index.copy()
        run_time.append(_time.time() - t1)

        if s > val_step:
            recent = np.array(err_record[(-val_step):])
            if np.sum((err_record[-val_step - 1] - recent) / err_record[-val_step - 1] >= min_gain) == 0:
                break

        pre_err = err

    index_record = index_record[: len(err_record), :].astype(int)

    if save_folder is not None:
        pd.DataFrame(index_record).to_csv(
            os.path.join(save_folder, f"{file_name}_index.txt"),
            header=False, index=False, sep="\t",
        )
        pd.DataFrame(
            np.transpose(np.vstack((err_record, np.array(range(len(err_record)))))),
            columns=["error", "steps"],
        ).to_csv(os.path.join(save_folder, f"{file_name}_error_and_step.txt"), index=False, sep="\t")

    return index_record, err_record, run_time


def generate_image_data(data, index, num_row, num_column, coord, image_folder=None, file_name=""):
    """Rearrange `data`'s columns per `index` and lay them out on a num_row x
    num_column grid at `coord`, returning a (num_row, num_column, n_samples) array
    with values in [0, 255] (higher original value -> higher pixel intensity)."""
    if isinstance(data, pd.DataFrame):
        data = data.values

    if image_folder is not None:
        os.makedirs(image_folder, exist_ok=True)

    data_2 = data.copy()[:, index]
    max_v = np.max(data_2)
    min_v = np.min(data_2)
    data_2 = 255 - (data_2 - min_v) / (max_v - min_v) * 255

    image_data = np.empty((num_row, num_column, data_2.shape[0]))
    image_data.fill(np.nan)
    for i in range(data_2.shape[0]):
        data_i = np.empty((num_row, num_column))
        data_i.fill(np.nan)
        data_i[coord] = data_2[i, :]
        data_i[np.where(np.isnan(data_i))] = 255
        image_data[:, :, i] = 255 - data_i  # high tabular value -> high pixel intensity

    return image_data
