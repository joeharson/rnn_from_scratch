"""Generate text from a trained character-level RNN.

The script works with an optional NumPy checkpoint containing the keys:
Wxh, Whh, bh, Why, by
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import config
from dataset import CharacterDataset
from layers import RNN, sample, softmax


def build_argument_parser():
	parser = argparse.ArgumentParser(description="Generate text with the RNN model.")
	parser.add_argument(
		"--checkpoint",
		default="",
		help="Optional .npz file containing model weights.",
	)
	parser.add_argument(
		"--prompt",
		default="",
		help="Seed text used to prime the hidden state before generation.",
	)
	parser.add_argument(
		"--length",
		type=int,
		default=300,
		help="Number of characters to generate.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1.0,
		help="Sampling temperature. Must be greater than zero.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=config.RANDOM_SEED,
		help="Random seed for reproducible sampling.",
	)
	return parser


def load_checkpoint(model, checkpoint_path):
	if not checkpoint_path:
		return False

	if not os.path.isfile(checkpoint_path):
		print(f"Checkpoint not found: {checkpoint_path}. Using random model weights.")
		return False

	try:
		data = np.load(checkpoint_path, allow_pickle=False)
		model.cell.load_parameters(data)
		return True
	except Exception as error:
		print(f"Could not load checkpoint {checkpoint_path}: {error}")
		print("Using random model weights instead.")
		return False


def normalize_prompt(dataset, prompt):
	prompt = prompt.lower()
	return "".join(ch for ch in prompt if ch in dataset.char_to_idx)


def encode_seed(dataset, prompt):
	if prompt:
		normalized = normalize_prompt(dataset, prompt)
		if normalized:
			return normalized

	return dataset.text[:1]


def prime_model(model, dataset, prompt):
	hidden_state = np.zeros((model.hidden_size, 1))
	current_vector = None

	for character in prompt:
		current_vector = dataset.one_hot(dataset.char_to_idx[character])
		hidden_state, _, _ = model.step(current_vector, hidden_state)

	if current_vector is None:
		current_vector = dataset.one_hot(dataset.char_to_idx[dataset.text[0]])

	return hidden_state, current_vector


def generate_text(model, dataset, prompt, length, temperature):
	if length < 0:
		raise ValueError("length must be greater than or equal to zero")

	if temperature <= 0:
		raise ValueError("temperature must be greater than zero")

	hidden_state, current_vector = prime_model(model, dataset, prompt)
	generated = [prompt]

	for _ in range(length):
		hidden_state, logits, _ = model.step(current_vector, hidden_state)
		probabilities = softmax(logits / temperature)
		next_index = sample(probabilities)
		next_character = dataset.decode_character(next_index)
		generated.append(next_character)
		current_vector = dataset.one_hot(next_index)

	return "".join(generated)


def main():
	parser = build_argument_parser()
	args = parser.parse_args()

	np.random.seed(args.seed)

	dataset = CharacterDataset(config.DATA_PATH)
	model = RNN(
		input_size=dataset.vocab_size,
		hidden_size=config.HIDDEN_SIZE,
		output_size=dataset.vocab_size,
	)

	checkpoint_path = args.checkpoint

	if not checkpoint_path and os.path.isfile("model.npz"):
		checkpoint_path = "model.npz"

	checkpoint_loaded = load_checkpoint(model, checkpoint_path)

	prompt = encode_seed(dataset, args.prompt)
	generated_text = generate_text(
		model=model,
		dataset=dataset,
		prompt=prompt,
		length=args.length,
		temperature=args.temperature,
	)

	if checkpoint_loaded:
		print(f"Loaded checkpoint: {checkpoint_path}")
	else:
		print("No checkpoint loaded; using the current model parameters.")

	print()
	print(generated_text)


if __name__ == "__main__":
	main()
