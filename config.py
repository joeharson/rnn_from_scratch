"""
Configuration file for the RNN project.
"""

# -----------------------------
# Dataset
# -----------------------------

DATA_PATH = "data/input.txt"

# -----------------------------
# Model
# -----------------------------

HIDDEN_SIZE = 256

SEQ_LENGTH = 40

# -----------------------------
# Training
# -----------------------------

LEARNING_RATE = 0.005

EPOCHS = 80

PRINT_EVERY = 100

GRADIENT_CLIP = 5.0

# -----------------------------
# Random Seed
# -----------------------------

RANDOM_SEED = 42