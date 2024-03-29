# testing with saved model
import pandas as pd
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

ZCR = "ZCR"
CHROMA_STFT = "CHROMA_STFT"
MFCC = "MFCC"
RMS = "RMS"
MEL_SPECTROGRAM = "MEL_SPECTROGRAM"
TEAGER_ENERGY_MFCC = "TEAGER_ENERGY_MFCC"

label_mapping = {
    '0': 'angry',
    '1': 'fear',
    '2': 'disgust',
    '3': 'sad',
    '4': 'happy',
    '5': 'neutral'
}


# extract zero cross rate feature
def extract_zcr_feature(data):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    return zcr


# extract Chroma_stft
def extract_chroma_stft(data, sample_rate):
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    return chroma_stft


# extract MFCC
def extract_mfcc(data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    return mfcc


# extract Root Mean Square Value
def extract_rms(data):
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    return rms


# extract mel spectrogram
def extract_mel_spectrogram(data, sample_rate):
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    return mel_spectrogram


# Teager Energy Based MFCC feature
def extract_teager_energy_mfcc(data, sample_rate):
    teager_energy = librosa.effects.preemphasis(data)
    teager_mfcc = np.mean(librosa.feature.mfcc(y=teager_energy, sr=sample_rate).T, axis=0)
    return teager_mfcc


# feature combinations
def extract_features(features, signal, sample_rate):
    result = np.array([])

    if ZCR in features:
        zcr = extract_zcr_feature(signal)
        result = np.hstack((result, zcr))

    if CHROMA_STFT in features:
        chroma_stft = extract_chroma_stft(signal, sample_rate)
        result = np.hstack((result, chroma_stft))

    if MFCC in features:
        mfcc_features = extract_mfcc(signal, sample_rate)
        result = np.hstack((result, mfcc_features))

    if RMS in features:
        rms = extract_rms(signal)
        result = np.hstack((result, rms))

    if MEL_SPECTROGRAM in features:
        mel_spectrogram = extract_mel_spectrogram(signal, sample_rate)
        result = np.hstack((result, mel_spectrogram))

    if TEAGER_ENERGY_MFCC in features:
        teager_mfcc = extract_teager_energy_mfcc(signal, sample_rate)
        result = np.hstack((result, teager_mfcc))

    return result


# feature extraction for prediction
def combine_features_for_prediction(path):
    signal, sample_rate = librosa.load(path)

    # without augmentation
    pure_result = extract_features([ZCR, CHROMA_STFT, MFCC, RMS, MEL_SPECTROGRAM, TEAGER_ENERGY_MFCC], signal,
                                   sample_rate)
    result = np.array(pure_result)

    return result


def extract_features_for_model_testing(features):
    x = [features]
    scaler = joblib.load('notebooks/scaler.bin')
    x_features = scaler.transform(x)
    x_features = np.expand_dims(x_features, axis=2)
    return x_features


def identify_emotion(path):
    features = combine_features_for_prediction(path)
    x_features = extract_features_for_model_testing(features)

    # load cnn model
    cnn_model = load_model('notebooks/new-tuned-ser-cnn-model.h5')
    predictions = cnn_model.predict(x_features)
    index = np.argmax(predictions)
    emotion = label_mapping.get(str(index))
    return emotion


ravdess_df = pd.read_csv('notebooks/crema_data_path_mapping.csv', index_col=0)

evaluation_path = []
evaluation_emotion = []
evaluation_identify = []

for path, emotion in zip(ravdess_df.Path, ravdess_df.Emotions):
    identified_emotion = identify_emotion(path)
    evaluation_path.append(path)
    evaluation_emotion.append(emotion)
    evaluation_identify.append(identified_emotion)

ravdess_evaluation = pd.DataFrame({
    'path': evaluation_path,
    'actual': evaluation_emotion,
    'predicted': evaluation_identify
})


ravdess_evaluation.to_csv('crema_evaluation_mapping.csv', index=False)
