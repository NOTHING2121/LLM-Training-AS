import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim  # which dim is feature

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_max = np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]  # dim feature
           
        # Reshape input to 2D
        if len(shape) > 2:
            A_flat = np.moveaxis(self.A, self.dim, -1).reshape(-1, C)  # (..., C, ...) -> (*, C) -> (-1, C)
            dLdA_flat = np.moveaxis(dLdA, self.dim, -1).reshape(-1, C)  # (..., C, ...) -> (*, C) -> (-1, C)

        # star = A_flat.shape[0]
        # dLdZ_flat = np.zeros_like(dLdA_flat)
        # for i in range(star):
        #     J = np.zeros((C, C))
        #     for m in range(C):
        #         for n in range(C):
        #             if m == n:
        #                 J[m, n] = A_flat[i, m] * (1 - A_flat[i, m])
        #             else:
        #                 J[m, n] = - A_flat[i, m] * A_flat[i, n]
        #     dLdZ_flat[i, :] = dLdA_flat[i, :] @ J

        dLdZ_flat = A_flat * (dLdA_flat - np.sum(dLdA_flat * A_flat, axis=1, keepdims=True))

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            dLdZ = dLdZ_flat.reshape(np.moveaxis(self.A, self.dim, -1).shape)  # (*, C)
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)  # (..., C, ...)

        return dLdZ
 

    