from part_a.item_response import *
from part_a.matrix_factorization import *
from part_a.knn import *
from sklearn.neighbors import NearestNeighbors



def svd_demean(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    return Q, s_root, Ut, item_means


def impute(train_matrix, theta, beta):
    # initialize imputed matrix
    imputed_matrix = np.zeros(train_matrix.shape)

    # iterate over each row in train_matrix
    for row in range(train_matrix.shape[0]):

        # iterate over each column in train_matrix
        for col in range(train_matrix.shape[1]):

            # if the value is missing, impute it
            if np.isnan(train_matrix[row][col]):
                imputed_matrix[row][col] = sigmoid(theta[row] - beta[col])

            # if the value is not missing, keep it
            else:
                imputed_matrix[row][col] = train_matrix[row][col]

    return imputed_matrix


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # load npz file
    train_matrix_imputed = np.load("imputed_matrix.npz")["arr_0"]


    k_lst = list(range(1, 200))
    matrix_lst = []
    acc_lst = []
    # for k in k_lst:
    # #impute using knn
    # nbrs = KNNImputer(n_neighbors=k)
    # # We use NaN-Euclidean distance measure.
    # imputed_matrix = nbrs.fit_transform(train_matrix.T).T
    mat = svd_reconstruct(train_matrix_imputed, 4)
    acc = sparse_matrix_evaluate(val_data, mat)

    acc_lst.append(acc)
    print("k = {}, Validation Accuracy: {}".format(4, acc))


    index = 0
    curr_acc = acc_lst[0]
    for i in range(5):
        if acc_lst[i] > curr_acc:
            index = i
            curr_acc = acc_lst[i]
    test_acc = sparse_matrix_evaluate(test_data, matrix_lst[index])
    print(f"For Part (a), my chosen k is {k_lst[index]}.")
    print(f"With this k, my Validation Accuracy is {curr_acc}.")
    print(f"With this k, my Test Accuracy is {test_acc}.")


if __name__ == "__main__":
    main()
