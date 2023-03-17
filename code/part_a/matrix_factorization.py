from utils import *
from scipy.linalg import sqrtm

import numpy as np

import matplotlib as m
import matplotlib.pyplot as p


def svd_reconstruct(matrix, k):
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

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(mean, data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - mean[data["user_id"][i]][data["question_id"][i]]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(mean, train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    #                                                                   #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    curr_mean = mean[n][q]
    c -= curr_mean
    curr_user = u[n]
    curr_question = z[q]
    uTz = np.dot(curr_user, curr_question)
    num_user, _ = u.shape
    num_question, _ = z.shape
    u_temp = []
    for i in range(num_user):
        u_i = u[i]
        grad = -lr * curr_question * (c - uTz)
        u_i -= grad
        u_temp.append(u_i)
    z_temp = []
    for j in range(num_question):
        z_j = z[j]
        grad = -lr * curr_user * (c - uTz)
        z_j -= grad
        z_temp.append(z_j)

    u = np.array(u_temp)
    z = np.array(z_temp)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_matrix, train_data, k, lr, num_iteration, calc_square=False):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param calc_square: state whether I want to calculate square error loss
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    #                                                                   #
    # Implement the function as described in the docstring.             #
    #####################################################################
    new_matrix = train_matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    # print(mu)
    lost_data = []
    for _ in range(num_iteration):
        u, z = update_u_z(mu, train_data, lr, u, z)
        if calc_square:
            lost_data.append(squared_error_loss(mu, train_data, u, z))
    mat = np.array(np.matmul(u, z.T) + mu)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, lost_data


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    #                                                                   #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_lst = [5, 10, 15, 20, 25]
    matrix_lst = []
    acc_lst = []
    for k in k_lst:
        matrix = svd_reconstruct(train_matrix, k)
        matrix_lst.append(matrix)
        acc_lst.append(sparse_matrix_evaluate(val_data, matrix))
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
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Part (b) Limitations of SVD
    # print("When using SVD, we filled in all the missing value NaN by 0,"
    #       " this means we are assuming that every student will answer incorrectly "
    #       "the question that they has not answered yet. This is a quite naive assumption"
    #       "because it is usually not the case that he will incorrectly answer all "
    #       "such questions. Thus, by using this assumption, we will definitely reduce "
    #       "the accuracy of the model because it no longer is as representative as "
    #       "the real situation.")

    #####################################################################
    #                                                                   #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    lr = 1e-2
    num_iteration = 2000
    k_lst = [2,5,10,15,20]
    matrix_lst = []
    acc_lst = []
    for k in k_lst:
        matrix, _ = als(train_matrix, train_data, k, lr, num_iteration)
        matrix_lst.append(matrix)
        acc = sparse_matrix_evaluate(val_data, matrix)

        acc_lst.append(acc)

    index = 0
    curr_acc = acc_lst[0]
    for i in range(5):
        if acc_lst[i] > curr_acc:
            index = i
            curr_acc = acc_lst[i]
    test_acc = sparse_matrix_evaluate(test_data, matrix_lst[index])

    _, train_result = als(train_matrix, train_data, k_lst[index], lr, num_iteration, True)

    p.figure()
    p.plot([i + 1 for i in range(num_iteration)], train_result, c='r')
    p.xlabel("Number of Iterations")
    p.suptitle('Squared Error Loss vs Number of Iterations')
    p.show()

    print(f"For Part (b), my chosen k is {k_lst[index]}.")
    print(f"With this k, my Validation Accuracy is {curr_acc}.")
    print(f"With this k, my Test Accuracy is {test_acc}.")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
