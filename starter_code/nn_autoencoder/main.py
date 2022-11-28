import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd

from dataloader import load_dataset
from model import Encoder, Decoder, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


def train_loop(dataloader, encoder, decoder, discrim, loss_fn, encoder_opt, decoder_opt, discrim_opt, generat_opt, adv=False):
    size = len(dataloader.dataset)
    for batch, row in enumerate(dataloader):
        # put data on GPU
        row = row.to(device)

        # Compute prediction and loss
        pred = decoder(encoder(row))
        loss = loss_fn(pred, row)

        # Backpropagation
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()

        if adv:
            # train discriminator
            encoder.eval()

            code = encoder(row)
            gauss = Variable(torch.Tensor(np.random.normal(0, 1, code.size()))).to(device)  # N(0, 1) prior for now
            dis_loss = -torch.mean(torch.log(discrim(gauss) + 1e-10) + torch.log(1 - discrim(code) + 1e-10))

            discrim_opt.zero_grad()
            dis_loss.backward()
            discrim_opt.step()

            encoder.train()

            # apply adversarial cost to generator
            code = encoder(row)
            dis_code = discrim(code)
            generator_loss = -torch.mean(torch.log(dis_code + 1e-10))

            generat_opt.zero_grad()
            generator_loss.backward()
            generat_opt.step()

        if batch % 3 == 0:
            loss, current = loss.item(), batch * len(row)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(train, test_frame, encoder, decoder, val=False):
    size = len(test_frame.dataset)
    correct = 0

    # keep track of sensitivity and specificity
    total_positive = 0
    total_negative = 0
    true_positive = 0
    true_negative = 0

    with torch.no_grad():
        prediction_set = {}
        # user_id, question_id, is_correct
        for i, row in enumerate(test_frame):
            question_id = row[0][0].item()
            user_id = row[0][1].item()
            is_correct = row[0][2].item()

            total_positive += is_correct
            total_negative += 1 - is_correct

            if user_id not in prediction_set:
                # get training row corresponding to user id
                train_row = train[user_id]

                # put data on GPU
                train_row = train_row.to(device)

                # compute prediction
                pred = decoder(encoder(train_row))
                pred = torch.sigmoid(pred)
                prediction_set[user_id] = pred

            user_pred = prediction_set[user_id]

            # compute accuracy
            if user_pred[question_id] > 0.5:
                pred = 1
            else:
                pred = 0
            if pred == is_correct:
                correct += 1
                if pred == 1:
                    true_positive += 1
                else:
                    true_negative += 1

    acc = correct / size
    sensitivity = true_positive / total_positive
    specificity = true_negative / total_negative

    err = 'Validation' if val else 'Test'

    print(f"{err} Error: \n "
          f"Accuracy: {(100 * acc):>0.1f}%, "
          f"Sensitivity: {(100 * sensitivity):>0.1f}%, "
          f"Specificity: {(100 * specificity):>0.1f}%\n")

    return acc


def main():
    adv = True
    train, valid, test = load_dataset()
    train_loader = DataLoader(train, batch_size=64, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=1, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=1, shuffle=True, pin_memory=True)

    # setup hyperparameters
    k_lst = [4, 6]
    lr_lst = [1e-3, 1e-5]
    num_epoch = 20

    # initialize models for each k
    best_acc = 0
    best_lr = 0
    best_k = 0
    best_epoch = 0
    for k in k_lst:
        encoder = Encoder(k).to(device)
        decoder = Decoder(k).to(device)
        discrim = Discriminator(k).to(device)

        for lr in lr_lst:
            encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
            decoder_opt = torch.optim.Adam(decoder.parameters(), lr=lr)
            generat_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
            discrim_opt = torch.optim.Adam(discrim.parameters(), lr=lr)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            for t in range(num_epoch):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loop(train_loader, encoder, decoder, discrim, criterion, encoder_opt, decoder_opt, discrim_opt, generat_opt, adv)
                val_acc = test_loop(train, valid_loader, encoder, decoder, val=True)
                print("Done!")

                # if validation accuracy is best so far, save model
                if val_acc > best_acc:
                    best_acc, best_lr, best_k, best_epoch = val_acc, lr, k, t
                    torch.save(encoder.state_dict(), "encoder.pt")
                    torch.save(decoder.state_dict(), "decoder.pt")

    print(f"Best model: k = {best_k}, lr = {best_lr}, epoch = {best_epoch}, accuracy = {best_acc}")
    # load best model and test on test set
    encoder = Encoder(best_k)
    encoder.load_state_dict(torch.load("encoder.pt"))
    encoder.to(device)
    decoder = Decoder(best_k)
    decoder.load_state_dict(torch.load("decoder.pt"))
    decoder.to(device)
    test_loop(train, test_loader, encoder, decoder)

    # initialize model with best k and search for best regularization parameter
    lam_lst = [0.01, 0.1, 1]
    best_lam = 0
    for lam in lam_lst:
        encoder = Encoder(best_k).to(device)
        decoder = Decoder(best_k).to(device)
        discrim = Discriminator(best_k).to(device)

        encoder_opt = torch.optim.Adam(encoder.parameters(), lr=best_lr, weight_decay=lam)
        decoder_opt = torch.optim.Adam(decoder.parameters(), lr=best_lr, weight_decay=lam)
        generat_opt = torch.optim.Adam(encoder.parameters(), lr=best_lr)
        discrim_opt = torch.optim.Adam(discrim.parameters(), lr=best_lr)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        for t in range(num_epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_loader, encoder, decoder, discrim, criterion, encoder_opt, decoder_opt, discrim_opt, generat_opt, adv)
            val_acc = test_loop(train, valid_loader, encoder, decoder, val=True)
            print("Done!")

            # if validation accuracy is best so far, save model
            if val_acc > best_acc:
                best_acc, best_lam = val_acc, lam
                torch.save(encoder.state_dict(), "encoder.pt")
                torch.save(decoder.state_dict(), "decoder.pt")

    print(f"Best model: k = {best_k}, lr = {best_lr}, lam = {best_lam}, accuracy = {best_acc}")
    # load best model and test on test set
    encoder = Encoder(best_k)
    encoder.load_state_dict(torch.load("encoder.pt"))
    encoder.to(device)
    decoder = Decoder(best_k)
    decoder.load_state_dict(torch.load("decoder.pt"))
    decoder.to(device)
    test_loop(train, test_loader, encoder, decoder)


if __name__ == "__main__":
    main()
