"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w: np.ndarray | None = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        n = X_train.shape[0]
        y_train = y_train.astype(int, copy=False)
        
        scores = X_train @ self.w
        correct = scores[np.arange(n), y_train][:, None]
        
        margins = scores - correct + 1.0
        margins[np.arange(n), y_train] = 0.0
        
        mask = (margins > 0).astype(X_train.dtype)
        mask[np.arange(n), y_train] = -np.sum(mask, axis=1)
        
        gradient = (X_train.T @ mask) / n
        gradient += 2.0 * self.reg_const * self.w

        return gradient
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        n_train, n_features = X_train.shape
        if self.w is None or self.w.shape != (n_features, self.n_class):
            self.w = 0.01 * np.random.uniform(-1.0, 1.0, size=(n_features, self.n_class))

        y_train = y_train.astype(int, copy=False)
        batch_size = min(256, n_train)

        for _ in range(self.epochs):
            indices = np.random.permutation(n_train)
            for start in range(0, n_train, batch_size):
                batch_idx = indices[start : start + batch_size]
                grad = self.calc_gradient(X_train[batch_idx], y_train[batch_idx])
                self.w -= self.lr * grad

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
        if self.w is None:
            raise ValueError("Need to initialize model weights.")

        scores = X_test @ self.w
        return np.argmax(scores, axis=1)
