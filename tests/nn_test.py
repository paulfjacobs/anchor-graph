import scipy.io
import numpy as np

from anchor_graph.anchor_graph import build_anchor_graph
from anchor_graph.anchor_graph import convert_z_to_w

from sklearn.neighbors import NearestNeighbors


# NOTE: one problem is that one of the nearest neighbors for each index is just going to be itself. At a distance of zero
# that is not what we want.  This is because the "the query set matches the training set".  Perhaps, one way of solving
# this problem would be to simply increase the number of neighbors we are looking for to +1 and then subtracting out the
# identity matrix.
# data is a 2D np.array
def nearest_neighbor_graph(data, number_of_neighbors):
    # adding +1 for the reason explained above
    nbrs = NearestNeighbors(n_neighbors=number_of_neighbors + 1, algorithm='ball_tree').fit(data)
    graph = nbrs.kneighbors_graph(data).toarray()

    # remove self-loops in graph
    return graph - np.identity(len(data))


# Get the average degree
def average_degree(adjacency):
    (n, n) = adjacency.shape
    avg_degree = 0.0
    for i in range(n):
        avg_degree += float(np.count_nonzero(adjacency[i])) / n
    return avg_degree


if __name__ == "__main__":
    matlab_data_map = scipy.io.loadmat('tests/USPS-MATLAB-train.mat')
    data_matrix = matlab_data_map['samples']

    # Load the anchor data. Of interest is the 1000 x 256 matrix of anchors called "anchor".  We transpose it so that it
    # is d x m.
    matlab_data_map = scipy.io.loadmat('tests/usps_anchor_1000.mat')
    anchor_matrix = matlab_data_map['anchor'].transpose()
    print "Finished loading."

    # calculate the weight graph Z (n x m)
    s = 3
    our_z = build_anchor_graph(data_matrix, anchor_matrix, s, 0)

    # convert to W; this is the adjacency matrix for the original samples
    # our_w[0,0] is occupied...that's not good
    our_w = convert_z_to_w(our_z)

    # NOTE: Is "s" the best choice to use for the NN? Maybe look at the average degree

    print "Average degree %f" % average_degree(our_w)


    # build the NN
    real_nn = nearest_neighbor_graph(data_matrix.transpose(), s)
    print "Got both."

    print "Average degree %f" % average_degree(real_nn)
