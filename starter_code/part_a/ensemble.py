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
IRT_iterations = 1500

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
    return new_data


def evaluate(theta1, beta1, theta2, beta2, theta3, beta3, test_data):
    total_correct = 0
    total = 0
    # Predict
    for i, u in enumerate(test_data["user_id"]):
        total += 1

        prediction = 0.
        # Let them predict
        q = test_data["question_id"][i]
        x1 = (theta1[u] - beta1[q]).sum()
        p_a1 = irt.sigmoid(x1)
        prediction += p_a1

        x2 = (theta2[u] - beta2[q]).sum()
        p_a2 = irt.sigmoid(x2)
        prediction += p_a2

        x3 = (theta3[u] - beta3[q]).sum()
        p_a3 = irt.sigmoid(x3)
        prediction += p_a3

        vote = prediction / 3
        if vote >= threshold and test_data["is_correct"][i]:
            total_correct += 1
        if vote < threshold and not test_data["is_correct"][i]:
            total_correct += 1

    return total_correct / total


def main():
    train_data, val_data, test_data = load_data()

    # Use IRT three times
    IRT_train1 = bootstrap(train_data)
    theta1, beta1, _, _, _ = irt.irt(IRT_train1, val_data, IRT_lr, IRT_iterations)

    IRT_train2 = bootstrap(train_data)
    theta2, beta2, _, _, _ = irt.irt(IRT_train2, val_data, IRT_lr, IRT_iterations)

    IRT_train3 = bootstrap(train_data)
    theta3, beta3, _, _, _ = irt.irt(IRT_train3, val_data, IRT_lr, IRT_iterations)

    # Predict on Validation Data
    valid_acc = evaluate(theta1, beta1, theta2, beta2, theta3, beta3, val_data)
    print(f"My final validation accuracy is {valid_acc}")

    # Predict on Test Data
    test_acc = evaluate(theta1, beta1, theta2, beta2, theta3, beta3, test_data)
    print(f"My final test accuracy is {test_acc}")


if __name__ == "__main__":
    main()
