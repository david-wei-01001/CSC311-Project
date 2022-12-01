import torch
from torch.utils.data import DataLoader

from deep_autenc.dataloader import load_dataset
from model import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


def train_loop(dataloader, model, loss_fn, optimizer):
    total_loss = 0

    for batch, row in enumerate(dataloader):
        # put data on GPU
        row = row.to(device)

        # Compute prediction and loss
        pred = model(row)
        loss = loss_fn(pred, row)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def test_loop(train, test_frame, model, val=False):
    size = len(test_frame.dataset)
    correct = 0

    with torch.no_grad():
        prediction_set = {}
        # user_id, question_id, is_correct
        for i, row in enumerate(test_frame):
            question_id = row[0][0].item()
            user_id = row[0][1].item()
            is_correct = row[0][2].item()

            if user_id not in prediction_set:
                # get training row corresponding to user id
                train_row = train[user_id]

                # put data on GPU
                train_row = train_row.to(device)

                # compute prediction
                pred = model(train_row)
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

    acc = correct / size

    return acc


def main():
    train, valid, test = load_dataset()
    train_loader = DataLoader(train, batch_size=64, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=1, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=1, shuffle=True, pin_memory=True)

    # setup hyperparameters
    k_lst = [10]
    lr_lst = [1e-4]
    num_epoch = 500
    verbose = True

    print("Searching through the following hyperparameters:")
    print("k: ", k_lst)
    print("learning rate: ", lr_lst)
    print("num_epoch: ", num_epoch)

    # initialize models for each k
    best_acc = 0
    best_lr = 0
    best_k = 0
    best_epoch = 0
    for k in k_lst:
        model = AutoEncoder(k)
        model.to(device)

        for lr in lr_lst:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
            for t in range(num_epoch):
                loss = train_loop(train_loader, model, criterion, optimizer)
                val_acc = test_loop(train, valid_loader, model, val=True)\

                if verbose:
                    print("Epoch {}, Loss {}, Val Acc {}".format(t, loss, val_acc))

                # if validation accuracy is best so far, save model
                if val_acc > best_acc:
                    best_acc, best_lr, best_k, best_epoch = val_acc, lr, k, t
                    print(f"New best model found with k={k}, lr={lr}, epoch={t}. "
                          f"Total loss: {loss}. Validation accuracy: {val_acc}")
                    torch.save(model.state_dict(), "model.pt")

    print(f"Best model: k = {best_k}, lr = {best_lr}, epoch = {best_epoch}, accuracy = {best_acc}")
    # load best model and test on test set
    model = AutoEncoder(best_k)
    model.load_state_dict(torch.load("model.pt"))
    model.to(device)
    test_acc = test_loop(train, test_loader, model)
    print(f"Test accuracy: {test_acc}")

    # initialize model with best k and search for best regularization parameter
    lam_lst = [0.001, 0.01, 0.1, 1]
    best_lam = 0
    best_epoch = 0
    for lam in lam_lst:
        model = AutoEncoder(best_k)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=lam)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        for t in range(num_epoch):
            loss = train_loop(train_loader, model, criterion, optimizer)
            val_acc = test_loop(train, valid_loader, model, val=True)

            # if validation accuracy is best so far, save model
            if val_acc > best_acc:
                best_acc, best_lam, best_epoch = val_acc, lam, t
                print(f"New best model found with lam={lam}, epoch={t}. Loss: {loss}. "
                      f"Validation accuracy: {val_acc}")
                torch.save(model.state_dict(), "model.pt")

    print(f"Best model: k = {best_k}, lr = {best_lr}, lam = {best_lam}, epoch = {best_epoch}, accuracy = {best_acc}")
    # load best model and test on test set
    model = AutoEncoder(best_k)
    model.load_state_dict(torch.load("model.pt"))
    model.to(device)
    test_acc = test_loop(train, test_loader, model)
    print(f"Test accuracy: {test_acc}")


if __name__ == "__main__":
    main()
