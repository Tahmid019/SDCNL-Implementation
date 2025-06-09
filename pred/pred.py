from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa

app = Flask(__name__)
model = tf.keras.models.load_model('word_embedding3.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    y, sr = librosa.load(file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    prediction = model.predict(np.expand_dims(mfccs_processed, axis=0))
    return jsonify({'depression_probability': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
