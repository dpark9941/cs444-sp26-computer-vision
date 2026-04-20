"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H_1,..., H_{k-1}] with the number of neurons H_i in the hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            # TODO: (Extra Credit) You may set parameters for Adam optimizer here

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return np.dot(X, W) + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        de_dw = np.dot(X.T, de_dz)
        de_db = np.sum(de_dz, axis=0)
        de_dx = np.dot(de_dz, W.T)
        # TODO: implement me
        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
         # TODO: implement me
        return (X > 0).astype(float)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        # Here, X can be either the input to the sigmoid or the output of the
        # sigmoid, depending on your implementation.
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this. Your MSE loss should be implemented as:
        # MSE = $0.5 \frac{1}{NC} \sum \limits_{i=1}^N \sum \limits_{j=1}^C (y_{ij} - p_{ij})^2$
        # where y and p are of shape (N, C).
        return 0.5 * np.mean((y - p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return (p - y) / (y.shape[0] * y.shape[1])

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """ 

        self.inputs = X # For use later during backpropagation
        self.outputs = {}

        # TODO: implement me. You'll want to store the output of each layer in 
        # self.outputs as they will be used during backpropagation. You can use 
        # functions like self.linear and self.relu to help organize your code.
        
        last_output = X
        # layers 1 to num_layers - 1 are linear + relu
        for i in range(1, self.num_layers):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            # Linear layer
            h = self.linear(W, last_output, b)
            self.outputs["h" + str(i)] = h 
            # ReLU activation
            a = self.relu(h)
            # Store activation output
            self.outputs["a" + str(i)] = a
            last_output = a
        
        # Last layer is linear + sigmoid
        i = self.num_layers
        W = self.params["W" + str(i)]
        b = self.params["b" + str(i)]
        h = self.linear(W, last_output, b)
        self.outputs["h" + str(i)] = h
        
        out = self.sigmoid(h)
        self.outputs["a" + str(i)] = out

        return out

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss
        """

        self.gradients = {}
        
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.sigmoid_grad if it helps organize your code.
        upstream = self.mse_grad(y, self.outputs["a"+str(self.num_layers)])
        for i in range(1, self.num_layers+1)[::-1]:
            if i == self.num_layers:
                # last layer: linear + Sigmoid
                upstream = upstream * self.sigmoid_grad(self.outputs["h"+str(i)])
                de_dw, de_db, de_dx = self.linear_grad(self.params["W"+str(i)], self.outputs["a"+str(i-1)], upstream)
                upstream = de_dx
            else:
                # linear + ReLU
                upstream = upstream * self.relu_grad(self.outputs["h"+str(i)])
                de_dw, de_db, de_dx = self.linear_grad(self.params["W"+str(i)], self.outputs["a"+str(i-1)] if i > 1 else self.inputs, upstream)
                upstream = de_dx
            self.gradients["W"+str(i)] = de_dw
            self.gradients["b"+str(i)] = de_db

        return self.mse(y, self.outputs["a"+str(self.num_layers)])

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        
        if self.opt == 'SGD':
            # TODO: implement SGD optimizer here
            for key in self.params:
                if key in self.gradients:
                    self.params[key] -= lr * self.gradients[key]
            
        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            if not hasattr(self, "m"):
                self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
                self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
                self.t = 0

            self.t += 1

            for key in self.params:
                if key not in self.gradients:
                    continue

                gradient = self.gradients[key]

                self.m[key] = b1 * self.m[key] + (1 - b1) * gradient
                self.v[key] = b2 * self.v[key] + (1 - b2) * (gradient ** 2)

                m_hat = self.m[key] / (1 - b1 ** self.t)
                v_hat = self.v[key] / (1 - b2 ** self.t)

                self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            raise NotImplementedError
        
