
import networkx as nx
import numpy as np


def generate_graphs(sizes, dens_multiplier=2):
    ''' Generates Erdos-Renyi graphs with given size and densities.

    Parameters:

        sizes: List of integers
            Sizes of each mode of the product graph.

        dens_multiplier: List of floats
            Density multiplier of each mode. Def: 2 for connected graphs.

    Outputs:

        Phi: List of graph Laplacians.
    '''

    n = len(sizes)
    if len(dens_multiplier) < n:
        dens_multiplier = [dens_multiplier for _ in range(n)]

    # List of graphs for each mode
    G = [nx.erdos_renyi_graph(sizes[i],
         dens_multiplier[i]*np.log(sizes[i])/sizes[i]) for i in range(n)]
    # Graph Laplacians of these graphs
    return [nx.laplacian_matrix(G[i]).todense() for i in range(n)]
