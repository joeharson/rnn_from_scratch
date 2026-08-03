# 🧠 RNN From Scratch (NumPy)

> **Build a Recurrent Neural Network from first principles using only NumPy.**
>
> No PyTorch. No TensorFlow. No autograd. No hidden abstractions.
>
> Every forward pass, every gradient, every weight update is implemented manually to understand **how recurrent neural networks actually learn**.
 
---

## Why This Project Exists

Most deep learning tutorials begin with something like:

```python
model = nn.RNN(...)
loss.backward()
optimizer.step()
```

That's excellent for building applications.

It's not enough if your goal is to understand what is happening underneath.

This repository is built for people who want to move beyond calling APIs and start understanding the mathematics and engineering that power deep learning frameworks.

Every equation is translated into code.

Every gradient is computed manually.

Every parameter update is visible.

If you've ever wondered what `loss.backward()` actually does, this project answers that question.

---

# What You'll Learn

Instead of treating an RNN as a black box, you'll implement every important component yourself.

✔ Character-Level Language Modeling

✔ One-Hot Encoding

✔ Hidden State Computation

✔ Forward Propagation

✔ Softmax

✔ Cross-Entropy Loss

✔ Backpropagation Through Time (BPTT)

✔ Gradient Descent

✔ Gradient Clipping

✔ Text Generation

By the end, concepts used in PyTorch and TensorFlow will feel much less "magical."

---

# Project Structure

```
rnn_from_scratch/
│
├── model.py          # RNN architecture
├── dataset.py        # Text loading & preprocessing
├── utils.py          # Helper utilities
├── loss.py           # Cross-Entropy Loss
├── optimizer.py      # SGD Optimizer
├── train.py          # Training loop
├── predict.py        # Text generation
├── config.py         # Hyperparameters
├── README.md
│
├── data/
│   └── input.txt
│
├── model.npz         # Saved checkpoint (generated after training)
│
├── requirements.txt
├── Dockerfile
└── .venv/
```

---

# Learning Pipeline

```
Raw Text
    │
    ▼
Character Encoding
    │
    ▼
One-Hot Vectors
    │
    ▼
Forward Pass
    │
    ▼
Hidden State Update
    │
    ▼
Output Logits
    │
    ▼
Softmax
    │
    ▼
Cross Entropy Loss
    │
    ▼
Backpropagation Through Time
    │
    ▼
Gradient Descent
    │
    ▼
Updated Weights
```

---

# How an RNN Thinks

Unlike a standard neural network, an RNN remembers previous information through its hidden state.

```
Character
    │
    ▼
One-Hot Vector
    │
    ▼
      ┌─────────────┐
────▶ │   RNN Cell  │ ────▶ Prediction
      └─────────────┘
             ▲
             │
      Previous Hidden State
```

The hidden state acts as the model's memory.

Every new character updates that memory before predicting the next one.

---

# Frequently Asked Questions

## Why only NumPy?

Because NumPy exposes every matrix multiplication, gradient, and parameter update.

Nothing is hidden behind framework abstractions.

The goal is understanding—not convenience.

---

## Why use One-Hot Encoding?

Neural networks only understand numbers.

Each character is converted into a vector where one position is 1 and every other position is 0.

Example:

```
Vocabulary

a b c d

Character = c

[0 0 1 0]
```

This numerical representation allows the network to process text mathematically.

---

## Why Softmax?

The RNN produces **logits**, not probabilities.

Softmax converts those values into a probability distribution.

Example:

```
Logits

[2.1, 0.5, 3.8]

↓

Softmax

[0.14, 0.03, 0.83]
```

The probabilities now sum to **1**, making it possible to choose the most likely next character.

---

## Why Cross-Entropy Loss?

The model needs a way to measure how wrong its predictions are.

Cross-Entropy compares the predicted probabilities with the correct answer.

* Correct prediction → Low loss
* Incorrect prediction → High loss

Training is simply the process of reducing this loss over time.

---

## What is Backpropagation Through Time (BPTT)?

Standard backpropagation works for independent samples.

Text is sequential.

Each prediction depends on previous hidden states.

BPTT unfolds the RNN through time and sends gradients backward across every time step.

```
x₁ → h₁ → h₂ → h₃ → Prediction

          ▲
          │
      Gradients Flow Back
```

This allows the model to learn contextual relationships instead of isolated characters.

---

## Why do gradients explode?

Gradients are repeatedly multiplied while moving backward through time.

If those values become too large, updates become unstable and training diverges.

The common solution is:

**Gradient Clipping**

which limits gradient magnitude before updating the weights.

---

## Why do gradients vanish?

If gradients are repeatedly multiplied by numbers smaller than one, they rapidly shrink toward zero.

Eventually, earlier layers receive almost no learning signal.

This makes remembering long-range information extremely difficult.

This limitation eventually led to architectures such as **LSTMs**, **GRUs**, and later **Transformers**.

---

## Why learn RNNs when Transformers exist?

Because nearly every important sequence modeling concept originates here.

Understanding RNNs makes it much easier to understand:

* LSTMs
* GRUs
* Attention
* Encoder–Decoder Models
* Transformers
* Large Language Models

Think of RNNs as learning the mechanics before driving the race car.

---

## Is this project production-ready?

No.

This repository is intentionally educational.

Its purpose is to expose every internal operation rather than maximize speed or accuracy.

Modern production systems typically rely on Transformer-based architectures.

---

# Running the Project

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd rnn_from_scratch
```

---

## 2. Create a Virtual Environment

> **Important**
>
> The `.venv` folder is machine-specific.
> If you move this project to another computer, create a new virtual environment instead of copying the existing one.

```powershell
python -m venv .venv
```

---

## 3. Activate the Environment (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

```powershell
& .\.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

or

```powershell
python -m pip install numpy
```

---

## 5. Train the Model

```powershell
python train.py
```

Training creates:

```
model.npz
```

which stores the learned parameters.

---

## 6. Generate Text

```powershell
python predict.py
```

---

### Generate from a Prompt

```powershell
python predict.py --prompt "Hello" --length 200
```

---

### Load a Specific Checkpoint

```powershell
python predict.py --checkpoint model.npz --prompt "Once upon a time" --length 300
```

---

# What is `model.npz`?

`model.npz` is a NumPy checkpoint generated during training.

It stores the learned weights so they can be reused without retraining.

`predict.py` loads this file to generate text.

If the checkpoint does not exist, train the model first.

---

# Recommended Workflow

```
Create Dataset
        │
        ▼
Train Model
        │
        ▼
Save model.npz
        │
        ▼
Generate Text
        │
        ▼
Experiment & Improve
```

---

# Docker

Build the image:

```bash
docker build -t rnn-from-scratch .
```

Train the model:

```bash
docker run --rm -it rnn-from-scratch
```

Generate text:

```bash
docker run --rm -it rnn-from-scratch predict.py --checkpoint model.npz --prompt "hello" --length 100
```

---

# Future Improvements

This project intentionally stays minimal.

Possible next steps include:

* LSTM
* GRU
* Word Embeddings
* Attention Mechanism
* Transformer Architecture
* Beam Search
* Adam Optimizer
* Mini-Batch Training
* GPU Support
* Mixed Precision Training
* Learning Rate Scheduling

---

# Learning Philosophy

Deep learning frameworks are incredible productivity tools.

But true intuition comes from building the machinery yourself.

This repository isn't just about training a character-level language model.

It's about understanding why neural networks learn, how gradients flow, why optimization works, and what modern AI frameworks automate behind the scenes.

Once these ideas become intuitive, moving to PyTorch, TensorFlow, JAX, or building your own architectures becomes significantly easier.

Build it.

Break it.

Understand it.

Then build something even better.

---

# Acknowledgements

Inspired by educational implementations that prioritize first-principles learning, classical neural network literature, and the philosophy of understanding before abstraction.

---

# License

This project is released for educational and personal learning purposes.

Feel free to study the code, modify it, experiment with new ideas, and use it as a foundation for exploring deeper neural network architectures.
