from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib as m
import matplotlib.pyplot as p

import part_a.knn as knn

# def sigmoid(inputs):
#     """
#     Apply the Logistic Sigmoid Function to the input
#     """
#     return 1 / (1 + np.exp(- inputs))


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.encoder = nn.Sequential(
            nn.Linear(num_question, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, k),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_question),
            nn.Sigmoid(),
        )

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        #                                                                   #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)
    num_student = train_data.shape[0]
    loss_func = nn.MSELoss(reduction="none")

    train_costs = []
    valid_accuracy = []
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            loss = loss_func(output, target)
            # mask out the missing entries.
            mask = torch.isnan(train_data[user_id])
            loss = loss[0][~mask]
            loss = loss.sum()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_costs.append(train_loss)
        valid_accuracy.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return train_costs, valid_accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)



        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    #                                                                   #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    k_lst = [1, 2, 3, 4, 5]
    lam_lst = [0.001, 0.01, 0.1, 1]
    evaluate_out = []
    _, num_question = zero_train_matrix.shape

    # Set optimization hyperparameters.
    lr = 1e-3
    num_epoch = 100
    train_loss_lst = []
    valid_accuracy_lst = []
    model_lst = []
    # Part C
    for k in k_lst:
        # Set model hyperparameters.
        model = AutoEncoder(num_question, k)

        train_loss_temp, valid_accuracy_temp = train(model, lr, 0, train_matrix, zero_train_matrix,
                                                     valid_data, num_epoch)
        result = evaluate(model, zero_train_matrix, valid_data)
        train_loss_lst.append(train_loss_temp)
        valid_accuracy_lst.append(valid_accuracy_temp)
        model_lst.append(model)
        evaluate_out.append(result)
    best_k = 0
    curr_acc = evaluate_out[0]
    for i in range(len(evaluate_out)):
        if evaluate_out[i] > curr_acc:
            curr_acc = evaluate_out[i]
            best_k = i
    k_star = k_lst[best_k]
    print(f"My best k is {k_star}")

    # Part D
    model = model_lst[best_k]
    train_loss = train_loss_lst[best_k]
    valid_accuracy = valid_accuracy_lst[best_k]
    epoch_lst = [i + 1 for i in range(num_epoch)]
    p.plot(epoch_lst, train_loss)
    p.xlabel("Number of Epoch")
    p.suptitle('Training Costs Changes w.r.t. Epoch')
    p.show()
    p.plot(epoch_lst, valid_accuracy)
    p.xlabel("Number of Epoch")
    p.suptitle('Validation Accuracy Changes w.r.t. Epoch')
    p.show()
    test_result = evaluate(model, zero_train_matrix, test_data)
    print(f"The final test accuracy without regularization is {test_result}")

    # Part E
    model_lst = []
    evaluate_out = []
    for lamb in lam_lst:
        # Set model hyperparameters.
        model = AutoEncoder(num_question, k_star)
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        result = evaluate(model, zero_train_matrix, valid_data)
        evaluate_out.append(result)
        model_lst.append(model)
    best_lamb = 0
    curr_acc = evaluate_out[0]
    for i in range(len(evaluate_out)):
        if evaluate_out[i] > curr_acc:
            curr_acc = evaluate_out[i]
            best_lamb = i
    lambda_star = lam_lst[best_lamb]
    model = model_lst[best_lamb]
    valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print(f"With regularization, my final chosen lambda is {lambda_star},"
          f" my final validation accuracy is {valid_accuracy},\n"
          f" and my final test accuracy is {test_accuracy}.")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    m.use("TKAgg")
    main()
