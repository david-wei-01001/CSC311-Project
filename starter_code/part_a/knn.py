from sklearn.impute import KNNImputer
from utils import *
import matplotlib as m
import matplotlib.pyplot as p


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    #                                                                   #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    #                                                                   #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_lst = [1, 6, 11, 16, 21, 26]

    # User Based Fit
    acc_lst = []
    for k in k_lst:
        acc_lst.append(knn_impute_by_user(sparse_matrix, val_data, k))
    p.figure()
    p.scatter(k_lst, acc_lst, c='r')
    p.xlabel("k value")
    p.suptitle('Accuracy vs k Based on Users')
    p.show()
    k = k_lst[0]
    biggest_acc = acc_lst[0]
    for i in range(len(k_lst)):
        if acc_lst[i] > biggest_acc:
            biggest_acc = acc_lst[i]
            k = k_lst[i]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k)
    print(f"Predicting using Users, "
          f"the k* that has the highest performance on validation data is {k}")
    print(f"And it's corresponding test accuracy is {test_acc}")

    # Question Based Fit
    acc_lst = []
    for k in k_lst:
        acc_lst.append(knn_impute_by_item(sparse_matrix, val_data, k))
    p.figure()
    p.scatter(k_lst, acc_lst, c='r')
    p.xlabel("k value")
    p.suptitle('Accuracy vs k Based on Questions')
    p.show()
    k = k_lst[0]
    biggest_acc = acc_lst[0]
    for i in range(len(k_lst)):
        if acc_lst[i] > biggest_acc:
            biggest_acc = acc_lst[i]
            k = k_lst[i]
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k)
    print(f"Predicting using Questions, "
          f"the k* that has the highest performance on validation data is {k}")
    print(f"And it's corresponding test accuracy is {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    m.use("TKAgg")
    main()
