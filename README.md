# Musical Chord Classifier using TensorFlow

This repository contains a TensorFlow model for classifying musical chords played by an instrument. The model has been trained using a Jupyter notebook, and can be used to classify chords in .wav audio files.

### Contents
Jupyter notebook for training the chord detector model
Use_model.py: A script that can be used to run the model against a .wav file and print out the detected chord
Chord model in .h5 format
## Usage
To use the model to classify chords in your own audio files, you can use the Use_model.py script. Simply provide the path to your .wav file as an argument to the script, and it will print out the detected chord.

### Example:
python Use_model.py path/to/your/audio.wav

### Requirements
TensorFlow
NumPy
Matplotlib (for visualizations in the Jupyter notebook)
## Note
---
This repository is for educational purposes and is not intended for commercial use. The model is a starting point and may require further training or modifications to achieve desired results for specific use cases.
