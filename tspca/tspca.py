""" targeted subspace Principal Component Analysis (tsPCA) """

# Author: Ryoma Hattori <rhattori0204@gmail.com>
#
# License:

from sklearn import linear_model
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import numpy as np


def tsPCA(input, targets, time_range, n_dim, target_types, reg, pc_num, axis_type, input_test, targets_test):

    if reg == None:
        d_model = linear_model.LogisticRegression(solver='lbfgs', penalty='none', n_jobs=-1, multi_class='auto', fit_intercept=True, max_iter=10000)
        c_model = linear_model.LinearRegression(n_jobs=-1, fit_intercept=True)
    elif reg == 'L1':
        d_model = linear_model.LogisticRegressionCV(cv=5, Cs=10, solver='saga', penalty='l1', n_jobs=-1, refit=True, multi_class='auto', fit_intercept=True, max_iter=10000)
        c_model = linear_model.LassoCV(cv=5, n_alphas=10, n_jobs=-1, fit_intercept=True, max_iter=10000)
    elif reg == 'L2':
        d_model = linear_model.LogisticRegressionCV(cv=5, Cs=10, solver='lbfgs', penalty='l2', n_jobs=-1, refit=True, multi_class='auto', fit_intercept=True, max_iter=10000)
        c_model = linear_model.RidgeCV(cv=5, alphas=np.logspace(-3, 1, 10), fit_intercept=True)
    # c_model.fit(np.mean(Xseq_ready[:, :, :], axis=1), Cc).coef_.shape
    # d_model.fit(np.mean(Xseq_ready[:, :, :], axis=1), Cc).coef_.squeeze().shape

    # if targets.ndim == 1:
    #     targets = np.repeat(targets[:, np.newaxis], n_dim, axis=1)
    #     target_types = np.repeat(target_types, n_dim)
    if pc_num != -1:
        pca_preprocessing = PCA(n_components=pc_num)
        if len(time_range) == 1:
            pca_preprocessing.fit_transform(input[:, time_range[0], :])
        else:
            pca_preprocessing.fit_transform(np.mean(input[:, time_range, :], axis=1))
        input = np.moveaxis(np.matmul(pca_preprocessing.components_, np.moveaxis(input, 1, 2)), 1, 2)
        if input_test is not None:
            input_test = np.moveaxis(np.matmul(pca_preprocessing.components_, np.moveaxis(input_test, 1, 2)), 1, 2)

    if len(time_range) == 1:
        # predictors = sm.add_constant(input[:, time_range[0], :])
        predictors = input[:, time_range[0], :]
    else:
        # predictors = sm.add_constant(np.mean(input[:, time_range, :], axis=1))
        predictors = np.mean(input[:, time_range, :], axis=1)
    if target_types[0] == 'c':
        if axis_type == 'hattori':
            # ax = sm.OLS(targets[:, 0], predictors).fit().params
            ax = c_model.fit(predictors, targets[:, 0]).coef_.squeeze()
        elif axis_type == 'mante':
            ax = np.zeros(predictors.shape[1])
            for cell_id in range(predictors.shape[1]):
                ax[cell_id] = c_model.fit(predictors[:, cell_id, np.newaxis], targets[:, 0]).coef_.squeeze()
            decoding_mante = np.zeros((input.shape[0], n_dim, targets.shape[1]))
            decoder_L2 = linear_model.RidgeCV(cv=5, alphas=np.logspace(-3, 1, 10), fit_intercept=True)
            decoder_L2.fit(predictors, targets[:, 0])
            decoding_mante[:, 0, 0] = decoder_L2.predict(predictors).squeeze()
    elif target_types[0] == 'd':
        if axis_type == 'hattori':
            # ax = sm.Logit(targets[:, 0], predictors).fit().params
            ax = d_model.fit(predictors[targets[:, 0]!=0, :], targets[targets[:, 0]!=0, 0]).coef_.squeeze()
        elif axis_type == 'mante':
            ax = np.zeros(predictors.shape[1])
            for cell_id in range(predictors.shape[1]):
                ax[cell_id] = d_model.fit(predictors[targets[:, 0] != 0, cell_id, np.newaxis], targets[targets[:, 0] != 0, 0]).coef_.squeeze()
            decoding_mante = np.zeros((input.shape[0], n_dim, targets.shape[1]))
            decoder_L2 = linear_model.LogisticRegressionCV(cv=5, Cs=10, solver='lbfgs', penalty='l2', n_jobs=-1, refit=True, multi_class='auto', fit_intercept=True, max_iter=10000)
            decoder_L2.fit(predictors, targets[:, 0])
            decoding_mante[:, 0, 0] = decoder_L2.predict(predictors).squeeze()
    # u, s, vh = np.linalg.svd(ax[1:, np.newaxis])
    u, s, vh = np.linalg.svd(ax[:, np.newaxis])
    projected = np.matmul(input, u)

    subspace = np.zeros((input.shape[0], input.shape[1], n_dim, targets.shape[1]))
    subspace[: ,:, 0, 0] = projected[:, :, 0]
    if input_test is not None:
        projected_test = np.matmul(input_test, u)
        subspace_test = np.zeros((input_test.shape[0], input_test.shape[1], n_dim, targets.shape[1]))
        subspace_test[:, :, 0, 0] = projected_test[:, :, 0]
        if axis_type == 'mante':
            decoding_mante_test = np.zeros((input_test.shape[0], n_dim, targets_test.shape[1]))
            if len(time_range) == 1:
                predictors_test = input_test[:, time_range[0], :]
            else:
                predictors_test = np.mean(input_test[:, time_range, :], axis=1)
            decoding_mante_test[:, 0, 0] = decoder_L2.predict(predictors_test).squeeze()

    for target_id in range(targets.shape[1]):
        if target_id == 0:
            start_dim = 1
        else:
            start_dim = 0
        for dim_id in range(start_dim, n_dim):
            if len(time_range) == 1:
                # predictors = sm.add_constant(projected[:, time_range[0], 1:])
                predictors = projected[:, time_range[0], 1:]
            else:
                # predictors = sm.add_constant(np.mean(projected[:, time_range, 1:], axis=1))
                predictors = np.mean(projected[:, time_range, 1:], axis=1)
            if target_types[target_id] == 'c':
                if axis_type == 'hattori':
                    # ax = sm.OLS(targets[:, dim_id], predictors).fit().params
                    ax = c_model.fit(predictors, targets[:, target_id]).coef_.squeeze()
                elif axis_type == 'mante':
                    ax = np.zeros(predictors.shape[1])
                    for cell_id in range(predictors.shape[1]):
                        ax[cell_id] = c_model.fit(predictors[:, cell_id, np.newaxis], targets[:, target_id]).coef_.squeeze()
                    decoder_L2.fit(predictors, targets[:, target_id])
                    decoding_mante[:, dim_id, target_id] = decoder_L2.predict(predictors).squeeze()
            elif target_types[target_id] == 'd':
                if axis_type == 'hattori':
                    # ax = sm.Logit(targets[:, dim_id], predictors).fit().params
                    ax = d_model.fit(predictors[targets[:, target_id]!=0, :], targets[targets[:, target_id]!=0, target_id]).coef_.squeeze()
                elif axis_type == 'mante':
                    ax = np.zeros(predictors.shape[1])
                    for cell_id in range(predictors.shape[1]):
                        ax[cell_id] = d_model.fit(predictors[targets[:, target_id] != 0, cell_id, np.newaxis], targets[targets[:, target_id] != 0, target_id]).coef_.squeeze()
                    decoder_L2.fit(predictors, targets[:, target_id])
                    decoding_mante[:, dim_id, target_id] = decoder_L2.predict(predictors).squeeze()
            # u, s, vh = np.linalg.svd(ax[1:, np.newaxis])
            u, s, vh = np.linalg.svd(ax[:, np.newaxis])
            projected = np.matmul(projected[:, :, 1:], u)
            subspace[: ,:, dim_id, target_id] = projected[:, :, 0]
            if input_test is not None:
                if axis_type == 'mante':
                    if len(time_range) == 1:
                        predictors_test = projected_test[:, time_range[0], 1:]
                    else:
                        predictors_test = np.mean(projected_test[:, time_range, 1:], axis=1)
                    decoding_mante_test[:, dim_id, target_id] = decoder_L2.predict(predictors_test).squeeze()
                projected_test = np.matmul(projected_test[:, :, 1:], u)
                subspace_test[:, :, dim_id, target_id] = projected_test[:, :, 0]

    subspace_remained = projected[:, :, 1:]
    r = np.zeros((targets.shape[1], n_dim, 2))
    for target_id in range(targets.shape[1]):
        for dim_id in range(n_dim):
            if len(time_range) == 1:
                r[target_id, dim_id, 0], r[target_id, dim_id, 1] = pearsonr(subspace[:, time_range[0], dim_id, target_id], targets[:, target_id])
            else:
                r[target_id, dim_id, 0], r[target_id, dim_id, 1] = pearsonr(np.mean(subspace[:, time_range, dim_id, target_id], axis=1), targets[:, target_id])
    if input_test is not None:
        r_test = np.zeros((targets.shape[1], n_dim, 2))
        for target_id in range(targets.shape[1]):
            for dim_id in range(n_dim):
                if len(time_range) == 1:
                    r_test[target_id, dim_id, 0], r_test[target_id, dim_id, 1] = pearsonr(subspace_test[:, time_range[0], dim_id, target_id], targets_test[:, target_id])
                else:
                    r_test[target_id, dim_id, 0], r_test[target_id, dim_id, 1] = pearsonr(np.mean(subspace_test[:, time_range, dim_id, target_id], axis=1), targets_test[:, target_id])
    else:
        r_test = []

    if axis_type == 'mante':
        r_decoding_mante = np.zeros((targets.shape[1], n_dim, 2))
        for target_id in range(targets.shape[1]):
            for dim_id in range(n_dim):
                r_decoding_mante[target_id, dim_id, 0], r_decoding_mante[target_id, dim_id, 1] = pearsonr(decoding_mante[:, dim_id, target_id], targets[:, target_id])
        if input_test is not None:
            r_decoding_mante_test = np.zeros((targets.shape[1], n_dim, 2))
            for target_id in range(targets.shape[1]):
                for dim_id in range(n_dim):
                    r_decoding_mante_test[target_id, dim_id, 0], r_decoding_mante_test[target_id, dim_id, 1] = pearsonr(decoding_mante_test[:, dim_id, target_id], targets_test[:, target_id])
        else:
            r_decoding_test = []
            r_decoding_mante_test = []
    else:
        r_decoding_mante = []
        r_decoding_mante_test = []


    return subspace, subspace_remained, r, r_test, r_decoding_mante, r_decoding_mante_test
