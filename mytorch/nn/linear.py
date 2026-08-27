import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

        self.in_features = in_features
        self.out_features = out_features

    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass

        *star, _ = A.shape
        A_flat = A.reshape(-1, self.in_features)  # (X, Din)
        self.A = A_flat  # Store input for backward pass

        Z_flat = A_flat @ self.W.T + self.b  # (X, Dout)

        Z = Z_flat.reshape(*star, self.out_features)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        *star, _ = dLdZ.shape
        dLdZ_flat = dLdZ.reshape(-1, self.out_features)  # (*, Dout) -> (-1, Dout)

        # Compute gradients (refer to the equations in the writeup)
        dLdA_flat = dLdZ_flat @ self.W
        self.dLdW = dLdZ_flat.T @ self.A
        self.dLdb = np.sum(dLdZ_flat, axis=0)

        self.dLdA = dLdA_flat.reshape(*star, self.in_features)  # (-1, Dout) -> (*, Dout)
        
        # Return gradient of loss wrt input
        return self.dLdA
