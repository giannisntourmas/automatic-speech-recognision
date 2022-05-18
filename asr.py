import librosa
import librosa.display
import numpy as np
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def pre_processing(signal_data):
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, top_db=40)

    signal_zcr = librosa.feature.zero_crossing_rate(signal_filtered)
    zcr_average = np.mean(signal_zcr)

    sf.write("filtered.wav", signal_filtered, 16000)

    return signal_filtered


def remove_noise(signal_data):
    return nr.reduce_noise(signal_data, sr=16000)


def digits_segmetation(signal_nparray):
    # We reverse the signal nparray.
    signal_reverse = signal_nparray[::-1]

    frames = librosa.onset.onset_detect(y=signal_nparray, sr=16000)
    times = librosa.frames_to_time(frames, sr=16000)
    samples1 = librosa.frames_to_samples(frames)

    frames_reverse = librosa.onset.onset_detect(y=signal_reverse, sr=16000)
    times_reverse = librosa.frames_to_time(frames_reverse, sr=16000)
    for i in range(0, len(times_reverse) - 1):
        times_reverse[i] = 0.03 - times_reverse[i]
        i += 1

    times_reverse = sorted(times_reverse)

    i = 0
    while i < len(times_reverse) - 1:
        if times_reverse[i + 1] - times_reverse[i] < 1:
            times_reverse = np.delete(times_reverse, i)
            i -= 1
        i += 1

    i = 0
    while i < len(times) - 1:
        if times[i + 1] - times[i] < 1:
            times = np.delete(times, i + 1)
            frames = np.delete(frames, i + 1)
            samples1 = np.delete(samples1, i + 1)
            i = i - 1
        i = i + 1

    merged_times = [*times, *times_reverse]
    merged_times = sorted(merged_times)

    samples1 = librosa.time_to_samples(merged_times, sr=16000)

    return samples1


def valid_digits(signal_data, samples):
    count_digits = 0
    digit = {}

    for i in range(0, len(samples), 2):
        if len(samples) % 2 == 1 and i == len(samples) - 1:
            digit[count_digits] = signal_data[samples[i - 1]:samples[i]]
        else:
            digit[count_digits] = signal_data[samples[i]:samples[i + 1]]
        count_digits += 1

    return digit


def get_training_samples_signal():
    training_samples_signals = {}
    index = 0

    for i in range(10):
        for name in ["s1", "s2", "s3"]:
            training_samples_signals[index], _ = librosa.load("./training/" + str(i) + "_" + name + ".wav", sr=16000)
            index += 1
    
    return training_samples_signals


def filter_dataset_signal(signal_data):
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, top_db=40)

    return signal_filtered


def recognition(digits, signal_data, dataset):
    recognized_digits_array = []
    j = 0
    while j < len(digits):
        mfcc_digit = librosa.feature.mfcc(y=digits[j], sr=16000, hop_length=480, n_mfcc=13)
        mfcc_digit_mag = librosa.amplitude_to_db(abs(mfcc_digit))

        cost_matrix_new = []
        mfccs = []
        # 0-9 from training set
        for i in range(len(dataset)):
            # We basically filter the training dataset as well.
            dataset[i] = filter_dataset_signal(dataset[i].astype(float))

            # MFCC for each digit from the training set
            mfcc = librosa.feature.mfcc(y=dataset[i], sr=16000, hop_length=80, n_mfcc=13)
            # logarithm of the features ADDED
            mfcc_mag = librosa.amplitude_to_db(abs(mfcc))

            # apply dtw
            cost_matrix, wp = librosa.sequence.dtw(X=mfcc_digit_mag, Y=mfcc_mag)

            # make a list with minimum cost of each digit
            cost_matrix_new.append(cost_matrix[-1, -1])
            mfccs.append(mfcc_mag)

        # index of MINIMUM COST
        index_min_cost = cost_matrix_new.index(min(cost_matrix_new))
        recognized_digits_array.append(["s1", "s2", "s3"][index_min_cost])
        j += 1

    return recognized_digits_array


file_path = "./training/0_s1.wav"
signal1, sr = librosa.load(file_path, sr=16000)
print(f"Original audio duration: {librosa.core.get_duration(signal1)}")

pre_proceed_signal = pre_processing(signal1)
print(f'New signal sound duration after filtering: {librosa.core.get_duration(pre_proceed_signal)}')

samples = digits_segmetation(pre_proceed_signal)
# print(f"samples = {samples}")

digits_array = valid_digits(pre_proceed_signal, samples)
# print(f"digits_array = {digits_array}")

dataset_training_signals = get_training_samples_signal()
# print(f"dataset_training_signals = {dataset_training_signals}")

recognized_digits = recognition(digits=digits_array, signal_data=pre_proceed_signal, dataset=dataset_training_signals)

print("Digits Recognized: ", end="")
for y in digits_array:
    print(y)

plt.subplot(2, 2, 1)
plt.title("Original Waveform")
librosa.display.waveshow(signal1, sr)

plt.subplot(2, 2, 2)
plt.title("Filtered Waveform")
librosa.display.waveshow(pre_proceed_signal, sr)

y = librosa.stft(signal1)
y_to_db = librosa.amplitude_to_db(abs(y))
plt.subplot(2, 2, 3)
plt.title('Original Spectrograph')
librosa.display.specshow(y_to_db, x_axis='time', y_axis='hz')
plt.colorbar(format="%2.f dB")

y2 = librosa.stft(pre_proceed_signal)
y2_to_db = librosa.amplitude_to_db(abs(y2))
plt.subplot(2, 2, 4)
plt.title('Filtered Spectrograph')
librosa.display.specshow(y2_to_db, x_axis='time', y_axis='hz')
plt.colorbar(format="%2.f dB")
plt.tight_layout()

plt.show()
