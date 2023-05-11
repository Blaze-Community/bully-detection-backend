import librosa
import os
import tensorflow as tf
import numpy as np

target_labels = [ "non-violence","violence", "unsure" ]

UPLOAD_FOLDER = 'static/uploads/'

SAMPLING_RATE = 44100
MFCC_NUM = 20
MFCC_MAX_LEN = 2000

__model = None

def wav2mfcc(wave, max_len=MFCC_MAX_LEN, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE):
    mfcc = librosa.feature.mfcc(wave, n_mfcc=n_mfcc, sr=sr)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def transform_audio(filepath):
  wave, sr = librosa.load(filepath, mono=True, sr=SAMPLING_RATE)
  wave = wave[::3]
  feat = wav2mfcc(wave)
  return feat

def transform_data(X: list):
  X = np.array(X)
  feature_dim_1 = MFCC_NUM
  feature_dim_2 = MFCC_MAX_LEN
  channel = 1
  X = X.reshape(X.shape[0], feature_dim_1, feature_dim_2, channel)
  return X

def prediction(audio_path):

    X = []
    X.append(transform_audio(audio_path))
    #print("audio converted to mfcc feats")
    
    X_test = transform_data(X)
    #print("input ready")
    
    y = __model.predict(X_test)
    #print("prediction done")
    print(y)

    maxElement = np.amax(y)
    print(maxElement)
    result = np.where(y == maxElement)
    print(result, result[0], result[1])
    print(result[1][0])
    
    predicted_class_name = target_labels[result[1][0]]
    
    output  = []
    output.append({
        'action_predicted' : str(predicted_class_name),
        'confidence' : str(maxElement)
    })
    
    return output

def upload_audio(file):
    filename = "temp"
    audio_path = os.path.join("../" + UPLOAD_FOLDER, filename)
    file.save(audio_path)
    return audio_path

def load_saved_audio_artifacts():
    print("loading saved audio artifacts...start")
    
    global __model 
    if __model is None:
        __model = tf.keras.models.load_model('./artifacts/audio_model_new_labels.h5')
        
    print("loading saved audio artifacts...done")

if __name__ == '__main__':
    load_saved_audio_artifacts()
    print(prediction('../audios/disgusted_03-01-07-01-01-01-01.wav'))