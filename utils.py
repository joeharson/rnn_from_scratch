import numpy as np


def softmax(logits):
    """
    Numerically stable softmax.

    Parameters
    ----------
    logits : np.ndarray
        Shape: (output_size, 1)

    Returns
    -------
    np.ndarray
        Probabilities with the same shape.
    """

    shifted = logits - np.max(logits)

    exp_values = np.exp(shifted)

    probabilities = exp_values / np.sum(exp_values)

    return probabilities


def one_hot(index, vocab_size):
    """
    Create one-hot encoded vector.

    Example:
        index = 2
        vocab_size = 5

        =>
        [[0]
         [0]
         [1]
         [0]
         [0]]
    """

    vector = np.zeros((vocab_size, 1))

    vector[index, 0] = 1

    return vector


def clip_gradients(gradients, threshold=5.0):
    """
    Gradient clipping.

    Prevents exploding gradients in RNNs.

    Parameters
    ----------
    gradients : dict

    threshold : float
    """

    for key in gradients:
        np.clip(
            gradients[key],
            -threshold,
            threshold,
            out=gradients[key]
        )

    return gradients


def sample(probabilities):
    """
    Sample one index from probability distribution.

    Used during text generation.
    """

    probabilities = probabilities.ravel()

    return np.random.choice(
        len(probabilities),
        p=probabilities
    )