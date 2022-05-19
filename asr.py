from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def pre_processing(signal_data, name):
    # Remove the background noise from the audio file.
    signal_reduced_noise = nr.reduce_noise(signal_data, sr=16000)
    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, top_db=40)
    sf.write("filtered_{}.wav".format(name), signal_filtered, 16000)
    return signal_filtered


def filter_dataset_signal(signal_data):
    # Remove the background noise from the audio file.
    signal_reduced_noise = nr.reduce_noise(signal_data, sr=16000)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, top_db=40)

    return signal_filtered


def get_training_samples_signal():
    training_samples_signals = {}
    index = 0
    for i in range(10):
        for name in ["s1", "s2", "s3"]:
            training_samples_signals[index], _ = librosa.load("./training/" + str(i) + "_" + name + ".wav", sr=16000)
            index += 1

    return training_samples_signals


def recognition(train_data, digit):
    mfcc_digit = librosa.feature.mfcc(y=digit, sr=16000, hop_length=480, n_mfcc=13)
    mfcc_digit_mag = librosa.amplitude_to_db(abs(mfcc_digit))
    cost_matrix_new = []
    mfccs = []
    for index, value in enumerate(train_data):
        train_data[index] = filter_dataset_signal(train_data[index])
        # MFCC for each digit from the training set
        mfcc = librosa.feature.mfcc(y=train_data[index], sr=16000, hop_length=80, n_mfcc=13)
        # logarithm of the features ADDED
        mfcc_mag = librosa.amplitude_to_db(abs(mfcc))
        # apply dtw
        cost_matrix, wp = librosa.sequence.dtw(X=mfcc_digit_mag, Y=mfcc_mag)
        # MFCC for each digit from the training set
        mfcc = librosa.feature.mfcc(y=train_data[index], sr=16000, hop_length=80, n_mfcc=13)
        # logarithm of the features ADDED
        mfcc_mag = librosa.amplitude_to_db(abs(mfcc))

        # make a list with minimum cost of each digit
        cost_matrix_new.append(cost_matrix[-1, -1])
        mfccs.append(mfcc_mag)
    # index of MINIMUM COST
    index_min_cost = cost_matrix_new.index(min(cost_matrix_new))

    recognized_digit = tags[index_min_cost]

    return recognized_digit


tags = []
for i in range(10):
    for j in range(1, 4, 1):
        tags.append("{}_s{}.wav".format(i, j))

# input the .wav file
sound_file = AudioSegment.from_wav("sample-1.wav")
real = [3, 5, 7, 9, 0, 2, 4, 6, 8, 1]
cnt = 0
# split words on silence
# must be silent for at least half a second
# consider it silent if quieter than -30 dBFS
audio_chunks = split_on_silence(sound_file, min_silence_len=300, silence_thresh=-40)

test_data = []
# make new .wav file for each word in audio file
for i, chunk in enumerate(audio_chunks):
    out_file = "./splitAudio/chunk{0}.wav".format(i)
    # print("exporting", out_file)
    chunk.export(out_file, format="wav")
    file, sr = librosa.load(out_file, sr=16000)
    # print(f"Original audio duration: {librosa.core.get_duration(file)}")
    file = pre_processing(file, i)
    # print(f'New signal sound duration after filtering: {librosa.core.get_duration(file)}')
    training_data = get_training_samples_signal()
    # print(training_data)
    recognize_digits = recognition(training_data, file)
    if real[i] == int(recognize_digits[0]):
        cnt += 1
    print(f"Prediction = {recognize_digits[0]} Real = {real[i]}")


print(f"Accuracy: {cnt / len(real) * 100} %")


