import numpy as np
import random
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Ring_graph, Grid_graph, ER_graph, RingPlus_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
from utilities import monitor
import os


def bits_run(dataset, graph_type, n, iid=True, nt=10, cepoch=400, depoch=100, save_file=False,
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
    m = lr_0.b
    cstep_size = cstep_size / L
    # dstep_size = {
    #     'step': 'constant',
    #     'lr': 0.001 / L,
    # }

    dstep_size = {
        'step': 'constant',
        'lr': 1 / 8000,
    }

    #dstep_size = {
    #    'step': 'decreasing',
    #    'lr': [1, 200, 50],
    #}

    """
    Initializing variables
    """
    theta_c0 = np.random.normal(0, 1, p)
    theta_0 = np.tile(theta_c0, (n, 1))
    if graph_type == 'grid':
        UG = Grid_graph(n).undirected()
    if graph_type == 'ring':
        UG = Ring_graph(n).undirected()
    if graph_type == 'ringplus':
        random.seed(123)
        UG = RingPlus_graph(n).undirected()
    if graph_type == 'er':
        random.seed(123)
        UG = ER_graph(n).undirected()
    if graph_type == 'exponential':
        UG = Exponential_graph(n).undirected()
    B = Weight_matrix(UG).metroplis()

    # plt.figure(1)
    # G = nx.from_numpy_matrix(np.matrix(UG), create_using=nx.Graph)
    # layout = nx.circular_layout(G)
    # nx.draw(G, layout)
    # plt.savefig('res/ER20.pdf', format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')
    # plt.show()

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
        # # CSGD
        # theta_CSGD = dopt.CSGD(lr_0, dstep_size, int(depoch), theta_c0)
        # res_x_CSGD = error_lr_0.ctheta_gap_path(theta_CSGD)
        # sgd.append(res_x_CSGD)

        # # DSGD
        # theta_DSGD = dopt.DSGD(lr_0, B, dstep_size, int(depoch), theta_0)
        # res_x_DSGD = error_lr_0.theta_gap_path(theta_DSGD)
        # dsgd.append(res_x_DSGD)

        # # CRR
        # theta_CRR = dopt.CRR(lr_0, dstep_size, int(depoch), theta_c0)
        # res_x_CRR = error_lr_0.ctheta_gap_path(theta_CRR)
        # crr.append(res_x_CRR)

        # DGD-RR
        theta_DGD_RR_C = dopt.DSGD_RR_C(lr_0, B, dstep_size, int(depoch), theta_0)
        res_x_DGD_RR_C = error_lr_0.theta_gap_path(theta_DGD_RR_C)
        drr.append(res_x_DGD_RR_C)

        # DGD-RR, communicate every epoch
        theta_DGD_RR = dopt.DSGD_RR(lr_0, B, dstep_size, int(depoch * m ), theta_0)
        res_x_DGD_RR = error_lr_0.theta_gap_path(theta_DGD_RR)
        drr_c.append(res_x_DGD_RR)

    #crr_mean = np.sum(crr, axis=0) / nt
    dsgd_mean = np.sum(dsgd, axis=0) / nt
    drr_mean = np.sum(drr, axis=0) / nt
    #sgd_mean = np.sum(sgd, axis=0) / nt
    drr_c_mean = np.sum(drr_c, axis=0) / nt


    """
    theta 
    """
    mark_every = int(len(drr_mean) * 0.1)
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(1)
    bits_seq = np.arange(0, len(drr_mean)) * m
    bits_seq_com = np.arange(0, len(drr_c_mean))
    #plt.plot(bits_seq, dsgd_mean, '-dy', markevery=mark_every)
    plt.plot(bits_seq, drr_mean, '-^m', markevery=mark_every)
    #plt.plot(bits_seq, crr_mean, '-vb', markevery=mark_every)
    #plt.plot(bits_seq, sgd_mean, '-<g', markevery=mark_every)
    plt.plot(bits_seq_com, drr_c_mean, '-Hc', markevery=int(len(drr_c_mean) * 0.1))
    plt.grid(True)
    plt.yscale('log')
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1, 3), axis='x')
    plt.tick_params(labelsize='large', width=3)
    plt.xlabel('Communication rounds', fontproperties=font)
    plt.ylabel(r'$\frac{1}{n}\sum_{i = 1}^n\mathbb{E}||x_{i,t}^0 - x^*||^2$', fontsize=12)
    plt.legend(('D-RR', 'DPG-RR'), prop=font2)
    plt.savefig('res/' + dataset + '/figs/' + graph_type + str(n) + 'nt' + str(nt) + '_bits' + '_' + str(iid) + '.pdf',
                format='pdf', dpi=4000, bbox_inches='tight')
    plt.show()

    if save_file:
        """
        Save data
        """
        if dataset == 'mnist':
            file_DSGD = 'res/mnist/dsgd'
            file_CRR = 'res/mnist/crr'
            file_SGD = 'res/mnist/sgd'
            file_DRR = 'res/mnist/drr'
            file_DRR_c = 'res/mnist/drr_c'
        elif dataset == 'cifar10':
            file_DSGD = 'res/cifar10/dsgd'
            file_CRR = 'res/cifar10/crr'
            file_SGD = 'res/cifar10/sgd'
            file_DRR = 'res/cifar10/drr'
            file_DRR_c = 'res/cifar10/drr_c'

        #np.savetxt(file_DSGD + graph_type + str(n) + '_' + str(iid) + '.txt', dsgd_mean)
        #np.savetxt(file_CRR + graph_type + str(n) + '_' + str(iid) + '.txt', crr_mean)
        #np.savetxt(file_SGD + graph_type + str(n) + '_' + str(iid) + '.txt', sgd_mean)
        np.savetxt(file_DRR + graph_type + str(n) + '_' + str(iid) + '_bits' + '.txt', drr_mean)
        np.savetxt(file_DRR_c + graph_type + str(n) + '_' + str(iid) + '_bits' + '.txt', drr_c_mean)


if __name__ == '__main__':

    #bits_run(dataset='mnist', graph_type='exponential', n=8, iid=False, nt=5, cepoch=500, depoch=20, save_file=False,
    #          cstep_size=0.1, momentum=0.95)

    bits_run(dataset='mnist', graph_type='grid', n=16, iid=False, nt=2, cepoch=500, depoch=400, save_file=True,
             cstep_size=1, momentum=0.9)

