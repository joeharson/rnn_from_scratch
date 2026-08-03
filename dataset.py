import numpy as np
import re


class CharacterDataset:
    """
    Character-level dataset.

    Reads text and converts it into:
    - character vocabulary
    - character ↔ index mappings
    - one-hot vectors
    """

    def __init__(self, filename):

        # Read file
        with open(filename, "r", encoding="utf-8") as f:
            self.text = self._normalize_text(f.read())

        # Unique characters
        self.characters = sorted(list(set(self.text)))

        self.vocab_size = len(self.characters)

        # Character -> Integer
        self.char_to_idx = {
            ch: i
            for i, ch in enumerate(self.characters)
        }

        # Integer -> Character
        self.idx_to_char = {
            i: ch
            for i, ch in enumerate(self.characters)
        }

        print(f"Dataset Size : {len(self.text)} characters")
        print(f"Vocabulary   : {self.vocab_size}")

    def _normalize_text(self, text):

        text = text.lower()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[^a-z0-9\s.,;:'!?-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text + "\n"

    # ------------------------------------------------

    def text_length(self):

        return len(self.text)

    # ------------------------------------------------

    def encode_character(self, ch):

        return self.char_to_idx[ch]

    # ------------------------------------------------

    def decode_character(self, idx):

        return self.idx_to_char[idx]

    # ------------------------------------------------

    def one_hot(self, index):

        vector = np.zeros((self.vocab_size, 1))

        vector[index, 0] = 1

        return vector

    # ------------------------------------------------

    def encode_text(self, text):

        return [
            self.encode_character(ch)
            for ch in text
        ]

    # ------------------------------------------------

    def decode_indices(self, indices):

        return "".join(
            self.decode_character(i)
            for i in indices
        )

    # ------------------------------------------------

    def get_sequence(self, start, seq_length):

        """
        Returns

        inputs
        targets

        Example

        hello

        input:

        h e l l

        target:

        e l l o
        """

        inputs = self.text[
            start:
            start + seq_length
        ]

        targets = self.text[
            start + 1:
            start + seq_length + 1
        ]

        x = [
            self.one_hot(
                self.encode_character(ch)
            )
            for ch in inputs
        ]

        y = [
            self.encode_character(ch)
            for ch in targets
        ]

        return x, y