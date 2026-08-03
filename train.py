import numpy as np

from dataset import CharacterDataset
from layers import RNN, clip_gradients, softmax
from loss import CrossEntropyLoss
from optimizer import SGD

import config

np.random.seed(config.RANDOM_SEED)

# ------------------------------------
# Load Dataset
# ------------------------------------

dataset = CharacterDataset(config.DATA_PATH)

# ------------------------------------
# Create Model
# ------------------------------------

model = RNN(
    input_size=dataset.vocab_size,
    hidden_size=config.HIDDEN_SIZE,
    output_size=dataset.vocab_size
)

criterion = CrossEntropyLoss()

optimizer = SGD(config.LEARNING_RATE)

print()

print("Vocabulary Size :", dataset.vocab_size)
print("Hidden Size     :", config.HIDDEN_SIZE)
print()


def save_checkpoint(model, filename="model.npz"):

    np.savez(filename, **model.cell.parameters())

# ------------------------------------
# Training Loop
# ------------------------------------

for epoch in range(config.EPOCHS):

    epoch_loss = 0

    total_sequences = 0

    for start in range(
        0,
        dataset.text_length() - config.SEQ_LENGTH - 1,
        config.SEQ_LENGTH
    ):

        # --------------------------
        # Build training sequence
        # --------------------------

        x, targets = dataset.get_sequence(
            start,
            config.SEQ_LENGTH
        )

        # --------------------------
        # Forward Pass
        # --------------------------

        hidden_states, outputs, caches = model.forward(x)

        loss = 0

        output_gradients = []

        # --------------------------
        # Compute Loss
        # --------------------------

        for t in range(len(outputs)):
            probs = softmax(outputs[t])

            loss += criterion.forward(
                probs,
                targets[t]
            )

            dy = criterion.backward(
                probs,
                targets[t]
            )

            output_gradients.append(dy)

        epoch_loss += loss

        total_sequences += 1

        gradients = model.backward(output_gradients, caches)
        gradients = clip_gradients(gradients, config.GRADIENT_CLIP)
        optimizer.step(model, gradients)

    average_loss = epoch_loss / total_sequences

    print(
        f"Epoch {epoch+1}/{config.EPOCHS}"
        f" | Loss = {average_loss:.4f}"
    )

print()

save_checkpoint(model)

print("Training Finished.")