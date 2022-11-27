from matplotlib.lines import Line2D

from part_b.autoencoder import AutoEncoder
from utils import *
from torch.autograd import Variable
from part_b.dataloder import load_data

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib as m
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, class_weights):
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    loss_func = nn.MSELoss(reduction='none')

    train_costs = []
    valid_accuracy = []
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            inputs = inputs.to(device)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            loss = loss_func(output, target)
            # mask out the missing entries.
            mask = torch.isnan(train_data[user_id])
            loss = loss[0][~mask]

            # weight the loss
            #loss = loss * class_weights[~mask]

            loss = loss.sum() + lamb * model.get_weight_norm() / 2
            loss.backward()
#            plot_grad_flow(model.named_parameters())

            train_loss += loss.item()
            optimizer.step()

        valid_acc, true_positive, true_negative = evaluate(model, zero_train_data, valid_data)
        train_costs.append(train_loss)
        valid_accuracy.append(valid_acc)
        print(f"Epoch {epoch}: train loss {train_loss}, valid accuracy {valid_acc}, true positive {true_positive},"
              f" true negative {true_negative}")
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

    total_positive = 0
    total_negative = 0

    true_positive = 0
    true_negative = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        inputs = inputs.to(device)
        output = model(inputs).to(device)
        # output = nn.Sigmoid()(output)

        # count total positive and negative
        if valid_data["is_correct"][i] == 1:
            total_positive += 1
        else:
            total_negative += 1

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        # count true positive and true negative
        if guess == valid_data["is_correct"][i]:
            if valid_data["is_correct"][i] == 1:
                true_positive += 1
            else:
                true_negative += 1
            correct += 1

        total += 1
    return correct / float(total), true_positive / float(total_positive), true_negative / float(total_negative)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    train_matrix_imputed = np.load("imputed_matrix.npz")["arr_0"]
    train_matrix_imputed = torch.from_numpy(train_matrix_imputed).float()
    train_matrix_imputed = train_matrix_imputed.to(device)
    #####################################################################
    #                                                                   #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    k_lst = [4]
    lam_lst = [0.001, 0.01, 0.1, 1]
    evaluate_out = []
    _, num_question = zero_train_matrix.shape

    # compute per class weights over train_matrix
    class_weights = torch.zeros(train_matrix.shape[1])
    for i in range(train_matrix.shape[1]):
        # ignore nan values
        class_weights[i] = 1 / torch.sum(~torch.isnan(train_matrix[:, i]))

    # Set optimization hyperparameters.
    lr = 1e-5
    num_epoch = 108
    train_loss_lst = []
    valid_accuracy_lst = []
    model_lst = []
    # # Part C
    # for k in k_lst:
    #     # Set model hyperparameters.
    #     model = AutoEncoder(num_question, k)
    #     model.to(device)
    #
    #     train_loss_temp, valid_accuracy_temp = train(model, lr, 0, train_matrix_imputed, train_matrix_imputed,
    #                                                  valid_data, num_epoch, class_weights)
    #
    #     result = evaluate(model, train_matrix_imputed, valid_data)
    #     train_loss_lst.append(train_loss_temp)
    #     valid_accuracy_lst.append(valid_accuracy_temp)
    #     model_lst.append(model)
    #     evaluate_out.append(result)
    #     plt.show()
    #
    # best_k = 0
    # curr_acc = evaluate_out[0]
    # for i in range(len(evaluate_out)):
    #     if evaluate_out[i] > curr_acc:
    #         curr_acc = evaluate_out[i]
    #         best_k = i
    # k_star = k_lst[best_k]
    # print(f"My best k is {k_star}")
    #
    # # Part D
    # model = model_lst[best_k]
    # train_loss = train_loss_lst[best_k]
    # valid_accuracy = valid_accuracy_lst[best_k]
    # epoch_lst = [i + 1 for i in range(num_epoch)]
    # plt.plot(epoch_lst, train_loss)
    # plt.xlabel("Number of Epoch")
    # plt.suptitle('Training Costs Changes w.r.t. Epoch')
    # plt.show()
    # plt.plot(epoch_lst, valid_accuracy)
    # plt.xlabel("Number of Epoch")
    # plt.suptitle('Validation Accuracy Changes w.r.t. Epoch')
    # plt.show()
    # test_result = evaluate(model, train_matrix_imputed, test_data)
    # print(f"The final test accuracy without regularization is {test_result}")

    # Part E
    num_epoch = 200
    model_lst = []
    evaluate_out = []
    for lamb in lam_lst:
        # Set model hyperparameters.
        model = AutoEncoder(num_question, 4)
        model.to(device)
        train(model, lr, lamb, train_matrix_imputed, train_matrix_imputed, valid_data, num_epoch, class_weights)
        result = evaluate(model, train_matrix_imputed, valid_data)
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
    valid_accuracy = evaluate(model, train_matrix_imputed, valid_data)
    test_accuracy = evaluate(model, train_matrix_imputed, test_data)
    print(f"With regularization, my final chosen lambda is {lambda_star},"
          f" my final validation accuracy is {valid_accuracy},\n"
          f" and my final test accuracy is {test_accuracy}.")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
