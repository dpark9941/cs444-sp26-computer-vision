"""Perceptron model."""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.b = None 
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
            x_val: a numpy array of shape (N, D) containing validation data;
                N examples with D dimensions
            y_val: a numpy array of shape (N,) containing validation labels
        """
        # TODO: implement me
        if self.w is None:
            self.w = np.random.uniform(-1,1, (self.n_class, X_train.shape[1])) * 0.01
        if self.b is None:
            self.b = np.zeros(self.n_class)
        
        lr = self.lr
        decay_rate = 0.5

        test_accuracies = []
        val_accuracies = []
        scores = np.zeros(self.n_class)

        for epoch in range(self.epochs):
            # shuffle data orders
            permutation = np.random.permutation(X_train.shape[0])
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            print(f"Epoch {epoch+1}/{self.epochs}")
            for i in range(X_train.shape[0]):
                scores = np.dot(self.w, X_train[i]) + self.b
                y_pred = np.argmax(scores, axis=0)
                if y_train[i] == y_pred:
                    continue
                else:
                    self.w[y_train[i]] = self.w[y_train[i]] + lr * X_train[i]
                    self.w[y_pred] = self.w[y_pred] - lr * X_train[i]
                    self.b[y_train[i]] = self.b[y_train[i]] + lr
                    self.b[y_pred] = self.b[y_pred] - lr

            y_preds = self.predict(X_train)
            accuracy = np.mean(y_preds == y_train) *100
            test_accuracies.append(accuracy)

            # y_val_preds = self.predict(x_val)
            # val_accuracy = np.mean(y_val_preds == y_val) *100
            # val_accuracies.append(val_accuracy)

            lr = self.lr * np.exp(-epoch / decay_rate)
        
        plt.title("Training Acc")
        self.plot_accuracy(test_accuracies)
        # plt.title("Validation Acc")
        # self.plot_accuracy(val_accuracies)


    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        if X_test is None:
            return None
        scores = np.dot(self.w, X_test.T) + self.b.reshape(-1, 1)
        y_pred = np.argmax(scores, axis=0)
        return y_pred
    
    
    def plot_accuracy(self, accuracies: np.ndarray):
        """Plot the training accuracy over epochs."""
        plt.plot(range(1, self.epochs + 1), accuracies, marker='o')
        # plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()
        
