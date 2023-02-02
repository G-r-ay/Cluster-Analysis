from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio

Chord_model = load_model('Chord_model.h5')
Chord_model.summary()

def Prediction(wav_file):
    def load_wav_16k_mono(filename):
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents,desired_channels=1)
        wav = tf.squeeze(wav,axis=1)
        sample_rate = tf.cast(sample_rate,dtype =tf.int64)
        wav = tfio.audio.resample(wav,rate_in=sample_rate,rate_out=16000)
        return wav

    def preprocess(file_path): 
        wav = load_wav_16k_mono(file_path)
        wav = wav[:48000]
        zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav],0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram

    
    chord = tf.data.Dataset.list_files([wav_file])
    data = tf.data.Dataset.zip(chord)
    
    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1000)
    data = data.batch(16)
    data = data.prefetch(8)

    Chord_model = load_model('Chord_model.h5')
    predictions = Chord_model.predict(data)
    results = [1 if prediction > 0.5 else 0 for prediction in predictions]
    if results[0] == 1:
        results = 'Major'
    else:
        results= 'Minor'
    print(results)


Prediction('Audio_Files/Minor/Minor_334.wav')