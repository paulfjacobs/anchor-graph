# This algorithm is an implementation of the one described in this paper (and the subsequent matlab code):
# Wei Liu, Junfeng He, and Shih-Fu Chang, "Large Graph Construction for Scalable Semi-Supervised Learning,"
# International Conference on Machine Learning (ICML), Haifa, Israel, 2010.

import numpy as np

__author__ = 'pjacobs'


def apply_to_all_rows(data_matrix, column_to_apply_on, value=float("inf")):
    # NOTE: column_to_apply_on needs to be the same length as the # of rows
    (rows, cols) = data_matrix.shape
    for i in range(rows):
        data_matrix[i, int(column_to_apply_on[i])] = value

# Return a tuple of the min value in the passed in vector and the index where that value occurs
def min_vector(vector):
    min_value = float("inf")
    min_index = 0
    for i in range(len(vector)):
        if vector[i] < min_value:
            min_value = vector[i]
            min_index = i

    return min_value, min_index


# This will return a tuple of column vectors (values, indices).  The value in values[i] is the min value at the vector
# data_matrix[i, :] and the indices[i] is the index where that min value is in data_matrix[i,:]
def min_matrix(data_matrix):
    (rows, cols) = data_matrix.shape
    values = np.zeros(rows)
    indices = np.zeros(rows)
    for i in range(rows):
        min_value, min_index = min_vector(data_matrix[i,:])
        values[i] = min_value
        indices[i] = min_index

    return values, indices

# NOTE: I don't think we really need to square it; probably just being used to compare distances
def square_distance(matrix1, matrix2):
    """
    matrix1 (d x n)
    matrix2 (d x m)
    return a matrix (n x m) where [i,j] is going to be the squared distance between the point i-th point in matrix1 and
     the j-th point in matrix2.
    """
    # TODO: error check that the dimensions match

    # Get the distances
    (d, n) = matrix1.shape
    (d, m) = matrix2.shape

    # Matrix that shows the distance between two points
    distances = np.zeros((n, m))

    # For each point in the first matrix calculate the squared distance between it and each point in the other matrix
    for i in range(n):
        for j in range(m):
            distances[i, j] = np.linalg.norm(matrix1[:, i] - matrix2[:, j])**2

    return distances


def build_anchor_graph(data_matrix, anchor_matrix, closest_anchors, weight_flag, num_iterations=0):
    """
    data_matrix: is (d x n) matrix of the input data; where 'n' is the number of samples and 'd' is the dimension of
    each sample.
    anchor_matrix: is the (d x m) matrix of anchors; where 'm' is the number of anchors and 'd' is the dimension of each
    anchor.  The anchors are generally not samples in the data_matrix but are derived from it.
    closest_anchors must be less than the total number of anchors; this is 's' variable; it's how many closest anchors
    we look for each sample

    Return The Z matrix (weight) which is (n x m).  Represents the weighted connection between samples and anchors.
    """

    # Extract important parameters from the dimensions of the matrices
    (dimensions, num_anchors) = anchor_matrix.shape
    num_samples = data_matrix.shape[1]

    # The Z matrix. This matrix will define the weighted connections between the samples and the anchors.
    # The idea is that closer anchors will have heavier weights, for a given sample, than farther ones.
    weight_matrix = np.zeros((num_samples, num_anchors))

    # Calculate the pairwise squared distances; passing in two matrices; we are calculating the distance between every
    # sample and every anchor
    # NOTE: Can this be done simply in numpy? I see they have a pdist function but that is not going to take in two
    # separate matrices
    distances = square_distance(data_matrix, anchor_matrix)

    # Track the distances of al the closest anchors for each sample
    distances_closest_anchors = np.zeros((num_samples, closest_anchors))
    indices_closest_anchors = np.zeros((num_samples, closest_anchors))

    # We will want to find the 'closest_anchors' number of closest anchors for each point; both the value and the indices
    for i in range(0, closest_anchors):
        # For each row (sample) determine the min values and associated indices for the closest anchor
        # NOTE: we can probably find the top closest_anchors for each in one go
        min_values, min_indices = min_matrix(distances)
        distances_closest_anchors[:, i] = min_values
        indices_closest_anchors[:, i] = min_indices

        # Now we are going to effectively make sure that we do re-use the same anchors by setting those distances to
        # infinity for each
        # NOTE: faster way of doing this? so you can use an array for the indices into the matrix; that would probably
        # be much faster.
        apply_to_all_rows(distances, min_indices)

    # Apply the kernel
    if weight_flag == 0:
        # sigma = mean(val(:,s).^0.5);
        # We calculate "sigma" which is going to be the equal to average of the square root of the maximum of min value
        # for each sample.  The last column of "distances_closest_anchors" will be the furthest away of the closest
        # anchors.
        sigma = np.mean(distances_closest_anchors[:, -1]**0.5)

        # NOTE: Possible error?  Maybe missing a () around the sigma^2.  Otherwise the 1/1 is not needed.
        #val = exp(-val/(1/1*sigma^2));
        distances_closest_anchors = np.exp(-1*distances_closest_anchors/(sigma**2))

        #val = repmat(sum(val,2).^-1,1,s).*val;
        distances_closest_anchors = np.transpose(np.tile(np.sum(distances_closest_anchors, axis=1)**-1,
                                                         (closest_anchors, 1)))*distances_closest_anchors

    else:
        # TODO: Apply LAE
        pass

    # Now we need to set the Z matrix; the indices_closest_anchors has the same number of rows as Z but fewer columns;
    # the values in that matrix at [i,j] corresponds to the column we are setting in Z and the value we are setting
    # there is going to by [i,j] in the distances_closest_anchors matrix.
    # TODO: Use better indexing
    for i in range(num_samples):
        for j in range(closest_anchors):
            weight_matrix[i, int(indices_closest_anchors[i, j])] = distances_closest_anchors[i, j]

    return weight_matrix

def convert_z_to_w(Z):
    # NOTE: Formula I have for the W creation from Z -- in MATLAB -- W = Z*diag(sum(Z).^-1)*transpose(Z)
    # diag(V) returns a square diagonal matrix with the elements of vector V on the main diagonal
    # so sum(Z) will return a vector where each element in the vector is the sum of that associated column in Z.
    # then .^-1 does an element-wise 1/x on each x
    # Z is a (n x m) matrix and we want to return the (n x n) matrix, W.
    # In dimension terms we have (n x n) = (n x m) * (m x m) * (m x n)

    # sum of the columns; make sure that Z is a float type array; produces a vector of length "m"; one element per column
    column_vector_sum = np.sum(np.asarray(Z, dtype=np.float), axis=0)

    # take each element to the power of -1
    power_vector = np.power(column_vector_sum, -1)

    # now put this power vector as the diagonal of matrix of zeros; multiple by the transpose
    return Z.dot(np.diag(power_vector)).dot(Z.transpose())