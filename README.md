# Part B Guide to read our code
- We first use SVD + kNN to give predictions, and this process can be seen in the file knn/matrix_factor_knn.py
- Then we went on to use more hidden layers wit different activation function and MSE to train the autoencoder, and this can be seen in the Neural Network folder
- Finally, we used Adversarial Autoencoder to train our model, and this can be seen in the deep_autenc folder

## Detail of Neural Network Folder
- dataloader.py will load the data
- autoencoder.py is the autoencoder object
- modified.py is the main function to run

## Detail of deep_autenc Folder
- dataloader.py will load the data
- model.py is the autoencoder object
- main.py is the main function to run
