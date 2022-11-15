import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Ring_graph, Grid_graph, ER_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
from utilities import monitor
import os
import random


def multi_run(dataset, graph_type, n, dstep_size, iid=True, nt=10, cepoch=400, depoch=100, save_file=False,
              cstep_size=0.1, momentum=0.95):
    if dataset == 'mnist':
        if iid:
            lr_0 = LR_L2(n, limited_labels=False, balanced=True)
        else:
            lr_0 = LR_L2(n, limited_labels=True, balanced=True)
    elif dataset == 'cifar10':
        if iid:
            lr_0 = LR_L4(n, limited_labels=False, balanced=True)
        else:
            lr_0 = LR_L4(n, limited_labels=True, balanced=True)
    else:
        print('Please choose from mnist or cifar10.')

    p = lr_0.p  # dimension of the model
    L = lr_0.L  # L-smooth constant
    cstep_size = cstep_size / L

    """
    Initializing variables
    """
    random.seed(12345)
    theta_c0 = np.random.normal(0, 1, p)
    theta_0 = np.tile(theta_c0, (n, 1))
    if graph_type == 'grid':
        UG = Grid_graph(n).undirected()
    if graph_type == 'ring':
        UG = Ring_graph(n).undirected()
    if graph_type == 'er':
        random.seed(123)
        UG = ER_graph(n).undirected()
    if graph_type == 'exponential':
        UG = Exponential_graph(n).undirected()
    B = Weight_matrix(UG).metroplis()
    lambdas = LA.eigvals(B)
    lambdas = sorted(lambdas)
    graph_gap = np.abs(lambdas[n - 2])
    print('lambda = ', graph_gap)

    """
    Centralized solutions
    """
    # solve the optimal solution of Logistic regression
    _, theta_opt, F_opt = copt.CNGD(lr_0, cstep_size, momentum, cepoch, theta_c0)
    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # initialization to store total error for each run
    crr = []
    dsgd = []
    drr = []
    sgd = []
    drr_c = []

    for it in range(nt):
        """
        Decentralized Algorithms
        """
        print('Implementing', it + 1, 'trail.....')
        # CSGD
        theta_CSGD = dopt.CSGD(lr_0, dstep_size, int(depoch), theta_c0)
        res_x_CSGD = error_lr_0.ngrad_path(theta_CSGD)
        sgd.append(res_x_CSGD)

        # DSGD
        theta_DSGD = dopt.DSGD(lr_0, B, dstep_size, int(depoch), theta_0)
        res_x_DSGD = error_lr_0.ngrad_path(np.sum(theta_DSGD, axis=1) / n)
        dsgd.append(res_x_DSGD)

        # CRR
        theta_CRR = dopt.CRR(lr_0, dstep_size, int(depoch), theta_c0)
        res_x_CRR = error_lr_0.ngrad_path(theta_CRR)
        crr.append(res_x_CRR)

        # DGD-RR
        theta_DGD_RR_C = dopt.DSGD_RR_C(lr_0, B, dstep_size, int(depoch), theta_0)
        res_x_DGD_RR_C = error_lr_0.ngrad_path(np.sum(theta_DGD_RR_C, axis=1) / n)
        drr.append(res_x_DGD_RR_C)

    crr_mean = np.sum(crr, axis=0) / nt
    dsgd_mean = np.sum(dsgd, axis=0) / nt
    drr_mean = np.sum(drr, axis=0) / nt
    sgd_mean = np.sum(sgd, axis=0) / nt

    if save_file:
        """
        Save data
        """
        if dataset == 'mnist':
            file_DSGD = 'res/mnist/dsgd'
            file_CRR = 'res/mnist/crr'
            file_SGD = 'res/mnist/sgd'
            file_DRR = 'res/mnist/drr'
        elif dataset == 'cifar10':
            file_DSGD = 'res/cifar10/dsgd'
            file_CRR = 'res/cifar10/crr'
            file_SGD = 'res/cifar10/sgd'
            file_DRR = 'res/cifar10/drr'

        np.savetxt(file_DSGD + graph_type + str(n) + '_' + str(iid) + '.txt', dsgd_mean)
        np.savetxt(file_CRR + graph_type + str(n) + '_' + str(iid) + '.txt', crr_mean)
        np.savetxt(file_SGD + graph_type + str(n) + '_' + str(iid) + '.txt', sgd_mean)
        np.savetxt(file_DRR + graph_type + str(n) + '_' + str(iid) + '.txt', drr_mean)

    """
    theta 
    """
    mark_every = int(depoch * 0.1)
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(1)
    plt.plot(dsgd_mean, '-dy', markevery=mark_every)
    plt.plot(drr_mean, '-^m', markevery=mark_every)
    plt.plot(crr_mean, '-vb', markevery=mark_every)
    plt.plot(sgd_mean, '-<g', markevery=mark_every)
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(labelsize='large', width=3)
    plt.xlabel('Epoch', fontproperties=font)
    plt.ylabel(r'$||\nabla f(\bar{x}_t)||^2$', fontsize=12)
    plt.legend(('DSGD', 'D-RR', 'CRR', 'SGD'), prop=font2)
    plt.savefig('res/' + dataset + '/figs/' + graph_type + str(n) + 'nt' + str(nt) + dstep_size['step']
                + '_' + str(iid) + '.pdf',
                format='pdf', dpi=4000, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':

    dstep_size = {
         'step': 'constant',
         'lr': 1 / 550,
    }
    # test
    multi_run(dataset='cifar10', graph_type='exponential', dstep_size=dstep_size, iid=False, n=16, nt=1, cepoch=20,
                depoch=500, save_file=False, cstep_size=1, momentum=0.9)
    # multi_run(dataset='cifar10', graph_type='exponential', dstep_size=dstep_size, iid=False, n=16, nt=10, cepoch=20,
    #            depoch=500, save_file=True, cstep_size=1, momentum=0.9)
    #
    # multi_run(dataset='cifar10', graph_type='er', dstep_size=dstep_size, iid=False, n=16, nt=10, cepoch=20,
    #            depoch=500, save_file=True, cstep_size=1, momentum=0.9)
    #
    # multi_run(dataset='cifar10', graph_type='grid', dstep_size=dstep_size, iid=False, n=16, nt=10, cepoch=20,
    #            depoch=500, save_file=True, cstep_size=1, momentum=0.9)