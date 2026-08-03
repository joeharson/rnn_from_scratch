import numpy as np


class CrossEntropyLoss:

    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, probabilities, target):

        """
        probabilities : (output_size,1)

        target : integer index
        """

        p = np.clip(
            probabilities[target, 0],
            self.epsilon,
            1.0
        )

        loss = -np.log(p)

        return loss

    def backward(self, probabilities, target):

        """
        Gradient of

        Softmax + CrossEntropy

        Returns:

            dL/dy

        where y are logits.
        """

        grad = probabilities.copy()

        grad[target, 0] -= 1

        return grad