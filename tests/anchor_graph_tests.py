from nose.tools import *
import scipy.io
import numpy as np
from anchor_graph import anchor_graph


def setup():
    pass

def teardown():
    pass

# NOTE: Is there a better way?
def exhaustive_matrix_match(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        return False

    (n, m) = matrix1.shape
    for i in range(n):
        for j in range(m):
            if abs(matrix1[i, j] - matrix2[i, j]) > 0.000001:
                print "BREAK (i, j, matrix1, matrix2) = (%i, %i, %f, %f, %f) " % \
                      (i, j, matrix1[i, j], matrix2[i, j], abs(matrix1[i, j] - matrix2[i, j]))
                return False

    return True

def test_basic():
    print "test_basic"
    # TODO: cleanup the weird file paths
    matlab_data_map = scipy.io.loadmat('tests/USPS-MATLAB-train.mat')
    data_matrix = matlab_data_map['samples']

    # Load the anchor data. Of interest is the 1000 x 256 matrix of anchors called "anchor".  We transpose it so that it
    # is d x m.
    matlab_data_map = scipy.io.loadmat('tests/usps_anchor_1000.mat')
    anchor_matrix = matlab_data_map['anchor'].transpose()
    print "finished loading"''

    # calculate the weight graph Z (n x m)
    s = 3
    our_z = anchor_graph.anchor_graph(data_matrix, anchor_matrix, s, 0)

    # get the canonical Z matrix from the matlab code
    matlab_data_map = scipy.io.loadmat('tests/USPS-MATLAB-Anchor-Graph.mat')
    canonical_z = matlab_data_map['Z']
    print "got ours"

    # NOTE: Due to, what I assume are floating point rounding differences, the pure array_equal returns false
    #assert np.array_equal(canonical_z, our_z)
    assert exhaustive_matrix_match(our_z, canonical_z)
