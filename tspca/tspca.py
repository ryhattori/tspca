""" targeted subspace Principal Component Analysis (tsPCA) """

# Author: Ryoma Hattori <rhattori0204@gmail.com>
#
# License: MIT License

from sklearn import linear_model
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import numpy as np


def tsPCA(input, targets, time_range, n_dim, reg=None, preprocessing_pc=-1, input_test=None, targets_test=None, decoding=False):
    if reg == None:
        model = linear_model.LinearRegression(n_jobs=-1, fit_intercept=True)
    elif reg == 'L1':
        model = linear_model.LassoCV(cv=5, n_alphas=10, n_jobs=-1, fit_intercept=True, max_iter=10000)
    elif reg == 'L2':
        model = linear_model.RidgeCV(cv=5, alphas=np.logspace(-3, 1, 10), fit_intercept=True)

    n_target = targets.shape[1]

    if decoding == True:
        decoder_L2 = {}
        for target_id in range(n_target):
            decoder_L2[target_id] = linear_model.RidgeCV(cv=5, alphas=np.logspace(-3, 1, 10), fit_intercept=True)

    if preprocessing_pc != -1:
        pca_preprocessing = PCA(n_components=preprocessing_pc)
        if len(time_range) == 1:
            pca_preprocessing.fit_transform(input[:, time_range[0], :])
        else:
            pca_preprocessing.fit_transform(np.mean(input[:, time_range, :], axis=1))
        input = np.moveaxis(np.matmul(pca_preprocessing.components_, np.moveaxis(input, 1, 2)), 1, 2)
        if input_test is not None:
            input_test = np.moveaxis(np.matmul(pca_preprocessing.components_, np.moveaxis(input_test, 1, 2)), 1, 2)

    if len(time_range) == 1:
        # activity = sm.add_constant(input[:, time_range[0], :])
        activity = input[:, time_range[0], :]
    else:
        # activity = sm.add_constant(np.mean(input[:, time_range, :], axis=1))
        activity = np.mean(input[:, time_range, :], axis=1)
    ax_nonorthogonal = np.zeros((activity.shape[1], n_target))
    for cell_id in range(activity.shape[1]):
        ax_nonorthogonal[cell_id, :] = model.fit(targets, activity[:, cell_id]).coef_.squeeze()
    if decoding == True:
        decoded = np.zeros((input.shape[0], n_dim, n_target + 1))
        for target_id in range(n_target):
            decoder_L2[target_id].fit(activity, targets[:, target_id])
            decoded[:, 0, target_id] = decoder_L2[target_id].predict(activity).squeeze()
    u, s, vh = np.linalg.svd(ax_nonorthogonal)
    for target_id in range(n_target):
        if np.dot(ax_nonorthogonal[:, target_id], u[target_id, :]) < 0:
            u[target_id, :] = -u[target_id, :]
        else:
            u[target_id, :] = u[target_id, :]

    projected = np.matmul(input, u.T)

    subspace = np.zeros((input.shape[0], input.shape[1], n_dim, n_target + 1))
    subspace[:, :, 0, :-1] = projected[:, :, :n_target]
    if input_test is not None:
        projected_test = np.matmul(input_test, u.T)
        subspace_test = np.zeros((input_test.shape[0], input_test.shape[1], n_dim, n_target + 1))
        subspace_test[:, :, 0, :-1] = projected_test[:, :, :n_target]
        if decoding == True:
            decoded_test = np.zeros((input_test.shape[0], n_dim, n_target + 1))
            if len(time_range) == 1:
                activity_test = input_test[:, time_range[0], :]
            else:
                activity_test = np.mean(input_test[:, time_range, :], axis=1)
            for target_id in range(targets_test.shape[1]):
                decoded_test[:, 0, target_id] = decoder_L2[target_id].predict(activity_test).squeeze()
    else:
        subspace_test = []
        decoded_test = []

    for dim_id in range(1, n_dim):
        if len(time_range) == 1:
            activity = projected[:, time_range[0], n_target:]
        else:
            activity = np.mean(projected[:, time_range, n_target:], axis=1)
        ax_nonorthogonal = np.zeros((activity.shape[1], n_target))
        for cell_id in range(activity.shape[1]):
            ax_nonorthogonal[cell_id, :] = model.fit(targets, activity[:, cell_id]).coef_.squeeze()
        if decoding == True:
            for target_id in range(n_target):
                decoder_L2[target_id].fit(activity, targets[:, target_id])
                decoded[:, dim_id, target_id] = decoder_L2[target_id].predict(activity).squeeze()
        u, s, vh = np.linalg.svd(ax_nonorthogonal)
        for target_id in range(n_target):
            if np.dot(ax_nonorthogonal[:, target_id], u[target_id, :]) < 0:
                u[target_id, :] = -u[target_id, :]
            else:
                u[target_id, :] = u[target_id, :]
        projected = np.matmul(projected[:, :, n_target:], u.T)
        subspace[:, :, dim_id, :-1] = projected[:, :, :n_target]

        if input_test is not None:
            if decoding == True:
                if len(time_range) == 1:
                    activity_test = projected_test[:, time_range[0], n_target:]
                else:
                    activity_test = np.mean(projected_test[:, time_range, n_target:], axis=1)
                for target_id in range(targets_test.shape[1]):
                    decoded_test[:, dim_id, target_id] = decoder_L2[target_id].predict(activity_test).squeeze()
            projected_test = np.matmul(projected_test[:, :, n_target:], u.T)
            subspace_test[:, :, dim_id, :-1] = projected_test[:, :, :n_target]

    subspace_remained = projected[:, :, n_target:]
    if len(time_range) == 1:
        activity = subspace_remained[:, time_range[0], :]
    else:
        activity = np.mean(subspace_remained[:, time_range, :], axis=1)
    pca_nontask = PCA(n_components=n_dim)
    pca_nontask.fit(activity)
    subspace[:, :, :, -1] = np.matmul(subspace_remained, pca_nontask.components_.T)
    if len(time_range) == 1:
        subspace_var = np.var(subspace[:, time_range[0], :, :], axis=0)
        total_var = np.sum(np.var(input[:, time_range[0], :], axis=0))
    else:
        subspace_var = np.var(np.mean(subspace[:, time_range, :, :], axis=1), axis=0)
        total_var = np.sum(np.var(np.mean(input[:, time_range, :], axis=1), axis=0))
    if input_test is not None:
        subspace_remained_test = projected_test[:, :, n_target:]
        subspace_test[:, :, :, -1] = np.matmul(subspace_remained_test, pca_nontask.components_.T)
        if len(time_range) == 1:
            subspace_var_test = np.var(subspace_test[:, time_range[0], :, :], axis=0)
            total_var_test = np.sum(np.var(input_test[:, time_range[0], :], axis=0))
        else:
            subspace_var_test = np.var(np.mean(subspace_test[:, time_range, :, :], axis=1), axis=0)
            total_var_test = np.sum(np.var(np.mean(input_test[:, time_range, :], axis=1), axis=0))
    else:
        subspace_remained_test = []
        subspace_test = []
        subspace_var_test = []
        total_var_test = []

    r = np.zeros((n_target + 1, n_target, n_dim, 2))
    for target_ax_id in range(n_target + 1):
        for target_var_id in range(n_target):
            for dim_id in range(n_dim):
                if len(time_range) == 1:
                    r[target_ax_id, target_var_id, dim_id, 0], r[target_ax_id, target_var_id, dim_id, 1] = pearsonr(subspace[:, time_range[0], dim_id, target_ax_id], targets[:, target_var_id])
                else:
                    r[target_ax_id, target_var_id, dim_id, 0], r[target_ax_id, target_var_id, dim_id, 1] = pearsonr(np.mean(subspace[:, time_range, dim_id, target_ax_id], axis=1), targets[:, target_var_id])
    if input_test is not None:
        r_test = np.zeros((n_target + 1, n_target, n_dim, 2))
        for target_ax_id in range(n_target + 1):
            for target_var_id in range(n_target):
                for dim_id in range(n_dim):
                    if len(time_range) == 1:
                        r_test[target_ax_id, target_var_id, dim_id, 0], r_test[target_ax_id, target_var_id, dim_id, 1] = pearsonr(subspace_test[:, time_range[0], dim_id, target_ax_id], targets_test[:, target_var_id])
                    else:
                        r_test[target_ax_id, target_var_id, dim_id, 0], r_test[target_ax_id, target_var_id, dim_id, 1] = pearsonr(np.mean(subspace_test[:, time_range, dim_id, target_ax_id], axis=1), targets_test[:, target_var_id])
    else:
        r_test = []

    if decoding == True:
        r_decoded = np.zeros((n_target, n_dim, 2))
        for target_id in range(n_target):
            for dim_id in range(n_dim):
                r_decoded[target_id, dim_id, 0], r_decoded[target_id, dim_id, 1] = pearsonr(decoded[:, dim_id, target_id], targets[:, target_id])
        if input_test is not None:
            r_decoded_test = np.zeros((n_target, n_dim, 2))
            for target_id in range(n_target):
                for dim_id in range(n_dim):
                    r_decoded_test[target_id, dim_id, 0], r_decoded_test[target_id, dim_id, 1] = pearsonr(decoded_test[:, dim_id, target_id], targets_test[:, target_id])
        else:
            r_decoded_test = []
    else:
        r_decoded = []
        r_decoded_test = []

    return subspace, subspace_remained, r, r_decoded, subspace_var, total_var, subspace_test, subspace_remained_test, r_test, r_decoded_test, subspace_var_test, total_var_test

