import jax
import jax.numpy as jnp
from pybraille import convertText

from train import train_model, get_train_ds, decode_braille_binary
from model import model
from validation_model import predict_braille, encode_character_modified, validate_result
from braille_to_stl import braille_to_scad, scad_to_stl

import gradio as gr

def convert_to_braille_and_create_stl(user_input):
    # Main code execution
    train_ds = get_train_ds()
    trained_params = train_model(train_ds, 'model.pkl')

    encoded_chars = encode_character_modified(user_input)

    model_outputs = []
    for encoded_char in encoded_chars:
        # Reshape to [1, 1, 1] for [batch_size, sequence_length, features]
        reshaped_input = jnp.array([[[encoded_char]]], dtype=jnp.float32)
        pred_binary = predict_braille(model, trained_params, reshaped_input)
        decoded_char = decode_braille_binary(pred_binary)
        model_outputs.append(decoded_char)

    # Combine individual character predictions
    combined_model_output = ''.join(model_outputs)

    # Convert user input to Braille for validation
    correct_braille = convertText(user_input)

    # Validate the result
    result = validate_result(combined_model_output, correct_braille)

    braille_to_scad("output", result)
    scad_to_stl("output")
    return "output.stl file saved to Downloads folder."

with gr.Blocks() as demo:
    name = gr.Textbox(label="Text to convert")
    output = gr.Textbox(label="Output")
    greet_btn = gr.Button("Translate")
    greet_btn.click(fn=convert_to_braille_and_create_stl, inputs=name, outputs=output, api_name="translate")

demo.launch()

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()