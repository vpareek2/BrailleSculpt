import jax
import jax.numpy as jnp

from pybraille import convertText
from train import train_model, get_train_ds, encode_character, decode_braille_binary
from model import model

def predict_braille(model, trained_params, encoded_char):
    # The input encoded_char should already be in shape [1, 1, 1]
    pred, _ = model.apply(trained_params, None, encoded_char)
    # Apply sigmoid and round to get binary values
    pred_binary = jnp.round(jax.nn.sigmoid(pred))
    return pred_binary

# Modify the encode_character function to handle strings
def encode_character_modified(char: str):
    if len(char) > 1:  # If the input is a string
        return [ord(c) for c in char]  # Encode each character
    else:
        return ord(char)  # Single character

def validate_result(model_output, correct_braille):
    if model_output == correct_braille:
        return model_output
    else:
        return correct_braille
