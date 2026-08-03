"""Reusable neural network layers for the character-level RNN project.

This module keeps the math for the recurrent cell in one place so training,
prediction, and future refactors can share the same implementation.
"""

from __future__ import annotations

import numpy as np


def softmax(logits):
	"""Return a numerically stable softmax for column-vector logits."""

	logits = np.asarray(logits)
	shifted = logits - np.max(logits, axis=0, keepdims=True)
	exp_values = np.exp(shifted)
	return exp_values / np.sum(exp_values, axis=0, keepdims=True)


def sample(probabilities):
	"""Sample an index from a probability vector."""

	probabilities = np.asarray(probabilities).reshape(-1)
	total = probabilities.sum()

	if total <= 0:
		raise ValueError("Probability vector must have a positive sum.")

	probabilities = probabilities / total
	return np.random.choice(len(probabilities), p=probabilities)


def clip_gradients(gradients, threshold=5.0):
	"""Clip gradient tensors in-place to the provided threshold."""

	for key in gradients:
		np.clip(gradients[key], -threshold, threshold, out=gradients[key])

	return gradients


class RNNCell:
	"""Vanilla RNN cell with tanh hidden state update."""

	def __init__(self, input_size, hidden_size, output_size):

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
		self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
		self.bh = np.zeros((hidden_size, 1))
		self.Why = np.random.randn(output_size, hidden_size) * 0.01
		self.by = np.zeros((output_size, 1))

	def forward(self, x, h_prev):
		"""Run one recurrent step.

		Parameters
		----------
		x : np.ndarray
			Column vector with shape (input_size, 1).
		h_prev : np.ndarray
			Previous hidden state with shape (hidden_size, 1).
		"""

		x = np.asarray(x)
		h_prev = np.asarray(h_prev)

		if x.shape != (self.input_size, 1):
			raise ValueError(
				f"Expected x shape {(self.input_size, 1)}, got {x.shape}."
			)

		if h_prev.shape != (self.hidden_size, 1):
			raise ValueError(
				f"Expected h_prev shape {(self.hidden_size, 1)}, got {h_prev.shape}."
			)

		input_part = self.Wxh @ x
		hidden_part = self.Whh @ h_prev
		total = input_part + hidden_part + self.bh
		h = np.tanh(total)
		y = self.Why @ h + self.by

		cache = {
			"x": x,
			"h_prev": h_prev,
			"input_part": input_part,
			"hidden_part": hidden_part,
			"total": total,
			"h": h,
			"y": y,
		}

		return h, y, cache

	def backward(self, dy, cache, dh_next=None):
		"""Backpropagate through one recurrent step."""

		x = cache["x"]
		h_prev = cache["h_prev"]
		h = cache["h"]

		dWhy = dy @ h.T
		dby = dy
		dh = self.Why.T @ dy

		if dh_next is not None:
			dh = dh + dh_next
		dtanh = (1 - h ** 2) * dh
		dWxh = dtanh @ x.T
		dWhh = dtanh @ h_prev.T
		dbh = dtanh
		dh_prev = self.Whh.T @ dtanh

		return {
			"dWxh": dWxh,
			"dWhh": dWhh,
			"dbh": dbh,
			"dWhy": dWhy,
			"dby": dby,
			"dh_prev": dh_prev,
		}

	def parameters(self):
		"""Return the cell parameters as a plain dictionary."""

		return {
			"Wxh": self.Wxh,
			"Whh": self.Whh,
			"bh": self.bh,
			"Why": self.Why,
			"by": self.by,
		}

	def load_parameters(self, parameters):
		"""Load parameters from a mapping or npz-like object."""

		required = ("Wxh", "Whh", "bh", "Why", "by")
		missing = [name for name in required if name not in parameters]

		if missing:
			raise ValueError(f"Missing parameter(s): {', '.join(missing)}")

		self.Wxh = np.asarray(parameters["Wxh"])
		self.Whh = np.asarray(parameters["Whh"])
		self.bh = np.asarray(parameters["bh"])
		self.Why = np.asarray(parameters["Why"])
		self.by = np.asarray(parameters["by"])


class RNN:
	"""A single-layer vanilla RNN over a sequence of one-hot vectors."""

	def __init__(self, input_size, hidden_size, output_size):

		self.hidden_size = hidden_size
		self.cell = RNNCell(input_size, hidden_size, output_size)

	def forward(self, sequence):
		"""Run the model over a sequence of input vectors."""

		h = np.zeros((self.hidden_size, 1))
		hidden_states = []
		outputs = []
		caches = []

		for x in sequence:
			h, y, cache = self.cell.forward(x, h)
			hidden_states.append(h)
			outputs.append(y)
			caches.append(cache)

		return hidden_states, outputs, caches

	def step(self, x, h_prev):
		"""Expose a single time-step forward pass for generation code."""

		return self.cell.forward(x, h_prev)

	def backward(self, output_gradients, caches):
		"""Run backpropagation through time over one sequence."""

		if len(output_gradients) != len(caches):
			raise ValueError("output_gradients and caches must have the same length")

		gradients = {
			"dWxh": np.zeros_like(self.cell.Wxh),
			"dWhh": np.zeros_like(self.cell.Whh),
			"dbh": np.zeros_like(self.cell.bh),
			"dWhy": np.zeros_like(self.cell.Why),
			"dby": np.zeros_like(self.cell.by),
		}

		dh_next = np.zeros((self.hidden_size, 1))

		for t in reversed(range(len(caches))):
			step_gradients = self.cell.backward(
				output_gradients[t],
				caches[t],
				dh_next,
			)

			gradients["dWxh"] += step_gradients["dWxh"]
			gradients["dWhh"] += step_gradients["dWhh"]
			gradients["dbh"] += step_gradients["dbh"]
			gradients["dWhy"] += step_gradients["dWhy"]
			gradients["dby"] += step_gradients["dby"]
			dh_next = step_gradients["dh_prev"]

		return gradients

	def save_parameters(self, filename):
		"""Save parameters in a NumPy checkpoint."""

		np.savez(filename, **self.cell.parameters())

