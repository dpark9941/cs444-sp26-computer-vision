"""Logistic regression model."""

import numpy as np
import matplotlib.pyplot as plt

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.
        return 1 / (1 + np.exp(-z))        


    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels

            X_val: a numpy array of shape (N, D) containing validation data;
                N examples with D dimensions
            y_val: a numpy array of shape (N,) containing validation labels
        """
        # TODO: implement me
        self.w = np.random.uniform(-1,1, (X_train.shape[1])) *0.01
        lr = self.lr

        test_accuracies = []
        val_accuracies = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            for i in range(X_train.shape[0]):
                z = np.dot(self.w, X_train[i])
                y_pred = self.sigmoid(z)
                error = y_pred - y_train[i]
                self.w -= lr * error * X_train[i]



            test_acc = np.mean(self.predict(X_train) == y_train) * 100
            test_accuracies.append(test_acc)
            # val_acc = np.mean(self.predict(X_val) == y_val) * 100
            # val_accuracies.append(val_acc)

            lr = self.lr * np.exp(-epoch / 0.5)
        
        plt.title("Training Acc")
        self.plot_accuracy(test_accuracies)
        # plt.title("Validation Acc")
        # self.plot_accuracy(val_accuracies)



    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        y_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            z = np.dot(self.w, X_test[i])
            y_pred[i] = 1 if self.sigmoid(z) >= self.threshold else 0

        return y_pred

    def plot_accuracy(self, accuracies: np.ndarray):
        """Plot the training accuracy over epochs."""
        plt.plot(range(1, self.epochs + 1), accuracies, marker='o')
        # plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()