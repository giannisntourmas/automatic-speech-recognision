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
    signal_reduced_noise = nr.reduce_noise(signal_data, sr=sample_rate)
    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, top_db=40)
    sf.write("filtered_{}.wav".format(name), signal_filtered, sample_rate)
    return signal_filtered


def filter_dataset_signal(signal_data):
    # Remove the background noise from the audio file.
    signal_reduced_noise = nr.reduce_noise(signal_data, sr=sample_rate)
    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, top_db=40)
    return signal_filtered


def get_training_samples_signal():
    training_samples_signals = {}
    index = 0
    for i in range(10):
        for name in ["s1", "s2", "s3"]:
            training_samples_signals[index], _ = librosa.load("./training/" + str(i) + "_" + name + ".wav", sr=sample_rate)
            index += 1

    return training_samples_signals


def recognition(train_data, digit):
    mfcc_digit = librosa.feature.mfcc(y=digit, sr=sample_rate, hop_length=480, n_mfcc=13)
    mfcc_digit_mag = librosa.amplitude_to_db(abs(mfcc_digit))
    cost_matrix_new = []
    mfccs = []
    for index, value in enumerate(train_data):
        train_data[index] = filter_dataset_signal(train_data[index])
        # MFCC for each digit from the training set
        mfcc = librosa.feature.mfcc(y=train_data[index], sr=sample_rate, hop_length=80, n_mfcc=13)
        # logarithm of the features ADDED
        mfcc_mag = librosa.amplitude_to_db(abs(mfcc))
        # apply dtw
        cost_matrix, wp = librosa.sequence.dtw(X=mfcc_digit_mag, Y=mfcc_mag)
        # MFCC for each digit from the training set
        mfcc = librosa.feature.mfcc(y=train_data[index], sr=sample_rate, hop_length=80, n_mfcc=13)
        # logarithm of the features ADDED
        mfcc_mag = librosa.amplitude_to_db(abs(mfcc))

        # make a list with minimum cost of each digit
        cost_matrix_new.append(cost_matrix[-1, -1])
        mfccs.append(mfcc_mag)
    # index of MINIMUM COST
    index_min_cost = cost_matrix_new.index(min(cost_matrix_new))
    return tags[index_min_cost]


def create_plots(original, filtered, n):
    plt.figure(figsize=(12, 8), num=n)
    plt.subplot(2, 2, 1)
    plt.title("Original Waveform")
    librosa.display.waveshow(original, sr)

    plt.subplot(2, 2, 2)
    plt.title("Filtered Waveform")
    librosa.display.waveshow(filtered, sr)

    y = librosa.stft(original)
    y_to_db = librosa.amplitude_to_db(abs(y))
    plt.subplot(2, 2, 3)
    plt.title('Original Spectrograph')
    librosa.display.specshow(y_to_db, x_axis='time', y_axis='hz')
    plt.colorbar(format="%2.f dB")

    y2 = librosa.stft(filtered)
    y2_to_db = librosa.amplitude_to_db(abs(y2))
    plt.subplot(2, 2, 4)
    plt.title('Filtered Spectrograph')
    librosa.display.specshow(y2_to_db, x_axis='time', y_axis='hz')
    plt.colorbar(format="%2.f dB")
    plt.tight_layout()


tags = []
for i in range(10):
    for j in range(1, 4, 1):
        tags.append("{}_s{}.wav".format(i, j))

sample_rate = 8000
# input the .wav file
sound_file = AudioSegment.from_wav("sample-2.wav")
# real = [3, 5, 7, 9, 0, 2, 4, 6, 8, 1] # sample 1
real = [1, 3, 5] # sample 2
cnt = 0
# split words on silence
# must be silent for at least half a second
# consider it silent if quieter than -30 dBFS
# make new .wav file for each word in audio file
audio_chunks = split_on_silence(sound_file, min_silence_len=300, silence_thresh=-40)
asr = []
for i, chunk in enumerate(audio_chunks):
    out_file = "./splitAudio/word_{0}.wav".format(i)
    # print("exporting", out_file)
    chunk.export(out_file, format="wav")
    file, sr = librosa.load(out_file, sr=sample_rate)
    # print(f"Original audio duration: {librosa.core.get_duration(file)}")
    file_filtered = pre_processing(file, i)
    # print(f'New signal sound duration after filtering: {librosa.core.get_duration(file)}')
    training_data = get_training_samples_signal()
    # print(training_data)
    recognize_digits = recognition(training_data, file_filtered)
    asr.append(int(recognize_digits[0]))
    create_plots(file, file_filtered, i + 1)

    if real[i] == int(recognize_digits[0]):
        cnt += 1
    # print(f"Prediction = {recognize_digits[0]} Real = {real[i]}")

print(f"Real: {real}\nASR:  {asr}")
print(f"Accuracy: {cnt / len(real) * 100} %")

plt.show()




