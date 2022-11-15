import numpy as np
import matplotlib.pyplot as plt
from graph import Weight_matrix, Exponential_graph, Exponential_draw_graph, Ring_graph, Grid_graph, RingPlus_graph, ER_graph
import numpy.linalg as LA
import networkx as nx
import random
n = 16
UG = Exponential_draw_graph(n).undirected()
# B = Weight_matrix(UG).metroplis()

plt.figure(1)
G = nx.from_numpy_matrix(np.matrix(UG), create_using=nx.Graph)
layout = nx.circular_layout(G)
nx.draw(G, layout, node_size=200, width=1)
plt.savefig('res/exp' + str(n) + '.pdf', format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')

UG = Exponential_graph(n).undirected()
B = Weight_matrix(UG).metroplis()
lambdas = LA.eigvals(B)
lambdas = sorted(lambdas)
graph_gap = np.abs(lambdas[n - 2])
print('exp, 1 - lambda = ', 1 - graph_gap)

n = 16
plt.figure(2)
G = nx.grid_2d_graph(4,4)
pos = {(x,y):(y,-x) for x,y in G.nodes()}
nx.draw(G, pos=pos, node_size=200, width=1)
plt.savefig('res/grid' + str(n) + '.pdf', format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')
# plt.show()
UG = Grid_graph(n).undirected()
B = Weight_matrix(UG).metroplis()
lambdas = LA.eigvals(B)
lambdas = sorted(lambdas)
graph_gap = np.abs(lambdas[n - 2])
print('grid, 1 - lambda = ', 1 - graph_gap)

n = 16
random.seed(123)
UG = ER_graph(n).undirected()
# B = Weight_matrix(UG).metroplis()

plt.figure(3)
G = nx.from_numpy_matrix(np.matrix(UG), create_using=nx.Graph)
random.seed(123)
layout = nx.random_layout(G)
nx.draw(G, layout, node_size=200, width=1)
plt.savefig('res/ER' + str(n) + '.pdf', format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')
B = Weight_matrix(UG).metroplis()
lambdas = LA.eigvals(B)
lambdas = sorted(lambdas)
graph_gap = np.abs(lambdas[n - 2])
print('er, 1 - lambda = ', 1 - graph_gap)

# n = 16
# UG = Exponential_draw_graph(n).undirected()
# B = Weight_matrix(UG).metroplis()
# lambdas = LA.eigvals(B)
# lambdas = sorted(lambdas)
# graph_gap = np.abs(lambdas[n - 2])
# print('lambda = ', graph_gap)