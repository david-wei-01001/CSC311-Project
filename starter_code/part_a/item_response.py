from utils import *

import numpy as np

import matplotlib as m
import matplotlib.pyplot as p


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    #                                                                   #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        theta_i = theta[cur_user_id]
        beta_j = beta[cur_question_id]
        c_ij = data["is_correct"][i]
        log_like = c_ij * (theta_i - beta_j) - np.log(1 + np.exp(theta_i - beta_j))
        log_lklihood += log_like

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    #                                                                   #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_user = len(theta)
    num_question = len(beta)
    update_theta_i = [0. for _ in range(num_user)]
    update_beta_j = [0. for _ in range(num_question)]
    for i in range(len(data["is_correct"])):
        entry = data["is_correct"][i]

        curr_user = data["user_id"][i]
        curr_question = data["question_id"][i]
        update_theta_i[curr_user] += lr * (entry - sigmoid(theta[curr_user] - beta[curr_question]))
    for i in range(num_user):

        # Plus because we want to minimize negative log likelihood function as loss function,
        # that means we want to maximize log likelihood function, so we should follow the gradient.
        theta[i] += update_theta_i[i]

    for i in range(len(data["is_correct"])):
        entry = data["is_correct"][i]
        curr_user = data["user_id"][i]
        curr_question = data["question_id"][i]
        update_beta_j[curr_question] += lr * \
                                        (sigmoid(theta[curr_user] - beta[curr_question]) - entry)
    for j in range(num_question):
        beta[j] += lr * update_beta_j[j]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_users = 542
    num_questions = 1774
    theta = np.random.rand(num_users)
    beta = np.random.rand(num_questions)

    val_acc_lst = []
    neg_lld_train = []
    neg_lld_vald = []

    for num_iteration in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_valid = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_train.append(neg_lld)
        neg_lld_vald.append(neg_lld_valid)
        print("Number of iteration: {} \t NLLK: {} \t Score: {}"
              .format(num_iteration, neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
    return theta, beta, val_acc_lst, neg_lld_train, neg_lld_vald


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    #                                                                   #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 1500
    iter_lst = [i for i in range(iterations)]
    theta, beta, _, train_lld, valid_lld = irt(train_data, val_data, lr, iterations)
    p.figure()
    p.plot(iter_lst, [(-1) * item for item in train_lld], c='r')
    p.xlabel("number of iterations")
    p.suptitle('Training Log-Likelihood vs Number of Iterations')
    p.show()
    p.plot(iter_lst, [(-1) * item for item in valid_lld], c='b')
    p.xlabel("number of iterations")
    p.suptitle('Validation Log-Likelihood vs Number of Iterations')
    p.show()
    print(f"The final validation accuracy is {evaluate(data=val_data, theta=theta, beta=beta)}")
    print(f"The final test accuracy is {evaluate(data=test_data, theta=theta, beta=beta)}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    #                                                                   #
    # Implement part (d)                                                #
    #####################################################################
    q1_index = 0
    q2_index = 1
    q3_index = 2
    beta_1 = beta[q1_index]
    beta_2 = beta[q2_index]
    beta_3 = beta[q3_index]
    lst_q1 = [sigmoid(this_theta - beta_1) for this_theta in theta]
    lst_q2 = [sigmoid(this_theta - beta_2) for this_theta in theta]
    lst_q3 = [sigmoid(this_theta - beta_3) for this_theta in theta]
    p.scatter(theta, lst_q1, label="Question 1")
    p.scatter(theta, lst_q2, label="Question 2")
    p.scatter(theta, lst_q3, label="Question 3")
    p.legend()
    p.xlabel("theta")
    p.suptitle('Probability of Correct Response vs Theta')
    p.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    m.use("TKAgg")
    main()
