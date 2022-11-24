import random
from utils import *
import item_response as irt
import neural_network as n
from sklearn.impute import KNNImputer
import numpy as np
from scipy.sparse import csr_matrix
import torch

# Global Variables:
IRT_lr = 0.01
IRT_iterations = 450

Neural_k = 10
Neural_lr = 0.01
Neural_num_epoch = 100
Neural_lam = 0.001

knn_k = 11
# item is 21

threshold = 0.5


def load_data():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    return train_data, val_data, test_data


def bootstrap(data):
    size = len(data["user_id"])
    new_data = {"user_id": [], "question_id": [], "is_correct": []}
    for _ in range(size):
        index = random.randint(0, size - 1)
        new_data["user_id"].append(data["user_id"][index])
        new_data["question_id"].append(data["question_id"][index])
        new_data["is_correct"].append(data["is_correct"][index])
    sparseMatrix = csr_matrix((new_data["is_correct"],
                               (new_data["user_id"], new_data["question_id"])),
                              shape=(542, 1774)).toarray()
    return new_data, sparseMatrix


def evaluate(theta, beta, model, matrix, train_data, test_data, tensor_train):
    total_correct = 0
    total = 0
    # Predict
    for i, u in enumerate(test_data["user_id"]):
        total += 1

        # Let IRT predict
        q = test_data["question_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = irt.sigmoid(x)
        irt_predict = p_a >= 0.5

        # Let Neural Network predict
        model.eval()
        inputs = n.Variable(tensor_train[u]).unsqueeze(0)
        output = model(inputs)

        neural_predict = output[0][test_data["question_id"][i]].item() >= 0.5

        # Let kNN predict
        cur_user_id = test_data["user_id"][i]
        cur_question_id = test_data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            knn_predict = 1
        else:
            knn_predict = 0

        vote = ((irt_predict + neural_predict + knn_predict) / 3)
        if vote >= threshold and test_data["is_correct"][i]:
            total_correct += 1
        if vote < threshold and not test_data["is_correct"][i]:
            total_correct += 1

    return total_correct / total


def main():
    train_data, val_data, test_data = load_data()

    # IRT first
    IRT_train, _ = bootstrap(train_data)
    theta, beta, _, _, _ = irt.irt(IRT_train, val_data, IRT_lr, IRT_iterations)

    # Neural Network with Regularization second
    _, Neural_matrix = bootstrap(train_data)
    model = n.AutoEncoder(1774, Neural_k)
    zero_train_matrix = Neural_matrix.copy()
    zero_train_matrix[np.isnan(Neural_matrix)] = 0
    n.train(model, Neural_lr, Neural_lam, torch.FloatTensor(Neural_matrix),
            torch.FloatTensor(zero_train_matrix),
            val_data, Neural_num_epoch)

    # kNN with user the third
    _, knn_matrix = bootstrap(train_data)
    nbrs = KNNImputer(n_neighbors=knn_k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(knn_matrix)

    # Predict on Validation Data
    valid_acc = evaluate(theta, beta, model, mat, train_data, val_data,
                         torch.FloatTensor(zero_train_matrix))
    print(f"My final validation accuracy is {valid_acc}")

    # Predict on Test Data
    test_acc = evaluate(theta, beta, model, mat, train_data, test_data,
                        torch.FloatTensor(zero_train_matrix))
    print(f"My final test accuracy is {test_acc}")


if __name__ == "__main__":
    main()


