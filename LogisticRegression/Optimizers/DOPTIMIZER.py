################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
from numpy import linalg as LA


def DSGD_RR(prd, W, para, epoch, theta_0):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (para['lr'][2] * k + para['lr'][1])
    else:
        def lr(k):
            return para['lr']

    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    n_node = prd.n
    n_data_node = prd.data_distr[1]
    sample_matrix = np.zeros([n_node, n_data_node], dtype=np.int)
    W = (W + np.eye(n_node)) / 2
    for ep in range(epoch):
        # stepsize
        step = lr(ep)
        # shuffling
        for i in range(n_node):
            sample_matrix[i] = np.random.permutation(n_data_node)
            # go through all the data
        for t in range(n_data_node):
            grad = prd.networkgrad(theta, sample_matrix[:, t])
            theta = theta - step * grad
        theta = np.matmul(W, theta)
        ut.monitor('DSGD_RR', ep, epoch)
        theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch


# communicate every inner loop
def DSGD_RR_C(prd, W, para, epoch, theta_0):
    n_data_node = prd.data_distr[1]
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (para['lr'][2] * k + para['lr'][1])
    else:
        def lr(k):
            return para['lr']

    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    n_node = prd.n
    # n_data_node = prd.data_distr[1]
    sample_matrix = np.zeros([n_node, n_data_node], dtype=np.int)
    W = (W + np.eye(n_node)) / 2
    for ep in range(epoch):
        # stepsize
        step = lr(ep)
        # shuffling
        for i in range(n_node):
            sample_matrix[i] = np.random.permutation(n_data_node)
            # go through all the data
        for t in range(n_data_node):
            grad = prd.networkgrad(theta, sample_matrix[:, t])
            theta = np.matmul(W, theta - step * grad)
        ut.monitor('DSGD_RR_C', ep, epoch)
        theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch


def DSGD(prd, W, para, epoch, theta_0):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (para['lr'][2] * k + para['lr'][1])
    else:
        def lr(k):
            return para['lr']

    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    n_data_node = prd.data_distr[1]
    n_node = prd.n
    W = (W + np.eye(n_node)) / 2
    for ep in range(epoch):
        # stepsize
        step = lr(ep)
        for t in range(n_data_node):
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(theta, sample_vec)
            theta = np.matmul(W, theta - step * grad)
        ut.monitor('DSGD', ep, epoch)
        theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch


def CRR(prd, para, epoch, theta_0):
    n_data_node = prd.data_distr[1]
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (para['lr'][2] * k + para['lr'][1])
    else:
        def lr(k):
            return para['lr']

    n_node = prd.n
    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    sample_matrix = np.zeros([n_node, n_data_node], dtype=np.int)
    for ep in range(epoch):
        # stepsize
        step = lr(ep)
        # shuffling
        for i in range(n_node):
            sample_matrix[i] = np.random.permutation(n_data_node)
        # go through all the data
        for t in range(n_data_node):
            theta_re = np.tile(theta, (n_node, 1))
            grad = prd.networkgrad(theta_re, sample_matrix[:, t])
            grad_a = np.mean(grad, axis=0)
            theta = theta - step * grad_a
        ut.monitor('CRR', ep, epoch)
        theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch


def CSGD(prd, para, epoch, theta_0):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (para['lr'][2] * k + para['lr'][1])
    else:
        def lr(k):
            return para['lr']

    n_node = prd.n
    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    n_data_node = prd.data_distr[1]
    for ep in range(epoch):
        # stepsize
        step = lr(ep)
        # print(step)
        for t in range(n_data_node):
            theta_re = np.tile(theta, (n_node, 1))
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(theta_re, sample_vec)
            grad_a = np.mean(grad, axis=0)
            theta -= step * grad_a
        ut.monitor('CSGD', ep, epoch)
        theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch

