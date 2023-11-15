import functools
import json
import math
from typing import Tuple, TypeVar
import warnings
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
import plotnine as gg
import matplotlib.pyplot as plt

from model import model


# Type definitions for clarity
T = TypeVar('T')
Pair = Tuple[T, T]

# Setting up parameters for the dataset and training
repetitions = 500
BATCH_SIZE = 8
SEQ_LEN = 64
DATA_SIZE = 27 * repetitions
TRAIN_SIZE = int(0.75 * DATA_SIZE)
VALID_SIZE = DATA_SIZE - TRAIN_SIZE


warnings.filterwarnings('ignore')

# Read from a JSON file
with open('braille_translation.json', 'r') as file:
    data = json.load(file)

# Dictionary mappings from the JSON data
bin_to_braille = data['bin_to_braille']
eng_to_bin = data['eng_to_bin']

# Function to encode a single character to its ASCII value
def encode_character(char: str) -> int:
    return ord(char)

# Function to decode an ASCII value back to a character
def decode_character(code: int) -> str:
    return chr(code)

# Function to encode a Braille binary string to an array of integers
def encode_braille_binary(binary_str: str) -> np.ndarray:
    return np.array([int(bit) for bit in binary_str], dtype=np.int32)

# Function to decode an array of integers back to a Braille binary string
def decode_braille_binary(binary_array: np.ndarray) -> str:
    return ''.join(map(str, binary_array))

# Function to generate Braille data
def generate_braille_data(seq_len: int, data_size: int) -> Pair[np.ndarray]:
    english_chars = list(eng_to_bin.keys())
    eng_seqs = np.random.choice(english_chars, (data_size, seq_len))

    # Encoding English sequences
    vectorized_encode_char = np.vectorize(encode_character)
    encoded_eng_seqs = vectorized_encode_char(eng_seqs)
    encoded_eng_seqs = encoded_eng_seqs[:, :, np.newaxis]  # Shape [data_size, seq_len, 1]

    # Encoding Braille sequences
    braille_seqs = np.vectorize(eng_to_bin.get)(eng_seqs)
    vectorized_encode_braille = np.vectorize(encode_braille_binary, signature='()->(n)')
    encoded_braille_seqs = vectorized_encode_braille(braille_seqs)

    return encoded_eng_seqs, encoded_braille_seqs

# Function to split data into training and validation sets
def split_data(eng_seqs, braille_seqs, train_size, valid_size):
    train_x = eng_seqs[:train_size]
    train_y = braille_seqs[:train_size]

    valid_x = eng_seqs[train_size:train_size + valid_size]
    valid_y = braille_seqs[train_size:train_size + valid_size]

    return (train_x, train_y), (valid_x, valid_y)

# Dataset class for iterating over the data
class Dataset:
    """Iterator over a dataset, yielding batch_size elements at a time."""
    def __init__(self, xy: Pair[np.ndarray], batch_size: int):
        self._x, self._y = xy
        self._batch_size = batch_size
        self._num_batches = self._x.shape[0] // batch_size
        self._idx = 0

    def __next__(self) -> Pair[np.ndarray]:
        if self._idx >= self._num_batches:
            raise StopIteration

        start = self._idx * self._batch_size
        end = start + self._batch_size
        x, y = self._x[start:end], self._y[start:end]
        self._idx += 1
        return x, y

    def __iter__(self):
        return self


# Generate and split the data
eng_seqs, braille_seqs = generate_braille_data(SEQ_LEN, DATA_SIZE)
train, valid = split_data(eng_seqs, braille_seqs, TRAIN_SIZE, VALID_SIZE)

# Creating dataset objects
train_ds = Dataset(train, BATCH_SIZE)
valid_ds = Dataset(valid, BATCH_SIZE)

def get_train_ds():
    return train_ds


def train_model(train_ds: Dataset, model_save_path: 'model.pkl') -> hk.Params:
    """Initializes and trains a model on train_ds, then saves the final params."""
    rng = jax.random.PRNGKey(428)
    opt = optax.adam(1e-3)

    @jax.jit
    def loss(params, x, y):
        pred, _ = model.apply(params, None, x)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(pred, y))

    @jax.jit
    def update(step, params, opt_state, x, y):
        l, grads = jax.value_and_grad(loss)(params, x, y)
        grads, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return l, params, opt_state

    # Initialize state
    sample_x, _ = next(train_ds)
    params = model.init(rng, sample_x)
    opt_state = opt.init(params)

    # Training loop
    for step in range(1000):
        try:
            x, y = next(train_ds)
        except StopIteration:
            break
        train_loss, params, opt_state = update(step, params, opt_state, x, y)
        if step % 100 == 0:
            print(f"Step {step}: train loss {train_loss}")

    # Save the trained parameters
    with open(model_save_path, 'wb') as f:
        pickle.dump(params, f)

    return params

# Main function to start training
if __name__ == '__main__':
    trained_params = train_model(train_ds)
