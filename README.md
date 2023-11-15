# BrailleSculpt

An English to Brail translator that 3d Prints.

## Description

This project stemmed from my interest to learn more about the world of Deep Learning. I wanted to learn more about Long Short Term Memory (LSTM) Neural Networks. I never previously got the opportunity to learn to build with JAX and Haiku so I thought this is a good opportunity.

This project involves developing a machine learning application to convert text into Braille, a tactile writing system used by people who are visually impaired. The core of the project is a neural network model, likely implemented using JAX, a high-performance numerical computing library. The model is trained to predict Braille representations of text characters, and the training process is managed by a function named `train_model`. The application uses a modified `encode_character` function to convert input text into a format suitable for model prediction. Each character of the input text is encoded, reshaped, and then passed through the `predict_braille` function, which applies the trained neural network model to predict the Braille equivalent. The output predictions for each character are then combined and validated against the correct Braille translation using the `validate_result` function. Finally `run.py` puts it all together. 
