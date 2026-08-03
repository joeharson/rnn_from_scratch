import numpy as np


class SGD:

    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def step(self, model, gradients):

        """
        Update all model parameters.
        """

        model.cell.Wxh -= self.learning_rate * gradients["dWxh"]

        model.cell.Whh -= self.learning_rate * gradients["dWhh"]

        model.cell.bh -= self.learning_rate * gradients["dbh"]

        model.cell.Why -= self.learning_rate * gradients["dWhy"]

        model.cell.by -= self.learning_rate * gradients["dby"]