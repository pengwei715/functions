# TODO
# Find a pretrained model to do the speech diarization for call center conversation.
# The function should have the following steps:
# 1. Noise and reverberation
#    - remove the noise and reverberation from the audio signal
#    - time domain: noise and reverberation.
#    - frequency domain: SpecAugment
#    - the common methods are spectral subtraction, spectral masking, and spectral mapping

# 2. Feature extraction ( the common features are MFCC, PLP, and filter bank)
#    - extract the features from the audio signal to time-frequency domain features
#    - the common features are MFCC, PLP, and filter bank

# 3. Voice activity detection (VAD) (the common methods are energy-based, zero-crossing rate-based, and model-based)


# 4. Speaker segmentation (detect the speaker change points)
#    - each segment is assumed to contain speech from a signle speaker
#    - common approaches: Uniform segmentation / speaker change detection (each turn is a segment)
#    - speaker change detection ( window comparison, window classification, ASR-alike)

# 5. Speaker embedding (extract the speaker embedding for each speaker)
#    - LSTM, TDNN, transformer, conformer
#    - Loss functions: cross entropy, triplet, angular softmax, CosFace, TE2E/GE2E

# 6. Speaker clustering (cluster the speaker embeddings to get the speaker labels)
#    - cluster the per-segment speaker embeddings to get the speaker labels
#    - clustering is not classification.
#    - clustering result only makes sense inside of one audio file
#    - offline clustering needs to use. AHC, K-means++, Spectral clustering.

# For each of the following funcitons, we can give different options to the user to choose.


import noisereduce as nr
import librosa
import matplotlib.pyplot as plt
import numpy as np


# class that use the noisereduce package to reduce the noise
# https://pypi.org/project/noisereduce/
import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
import soundfile as sf


class NoiseReducer:
    def __init__(self, sr=None, prop_decrease=1.0, verbose=False, **kwargs):
        """
        Initialize the NoiseReducer class with various parameters.

        :param sr:            Sampling rate of the input signal.
        :param prop_decrease: The proportion to reduce the noise by (1.0 = 100%).
        :param verbose:       If True, produces verbose outputs like plots.
        :param kwargs:        Additional parameters for noise reduction including:
            - y_noise (np.ndarray): Noise signal to compute statistics over for non-stationary noise reduction.
            - stationary (bool, default=False): Whether to perform stationary or non-stationary noise reduction.
            - time_constant_s (float, default=2.0): Time constant in seconds for the non-stationary algorithm.
            - freq_mask_smooth_hz (int, default=500): Frequency range in Hz to smooth the mask over.
            - time_mask_smooth_ms (int, default=50): Time range in milliseconds to smooth the mask over.
            - thresh_n_mult_nonstationary (int, default=1): Used only in nonstationary noise reduction.
            - sigmoid_slope_nonstationary (int, default=10): Used only in nonstationary noise reduction.
            - n_std_thresh_stationary (int, default=1.5): Std. deviations above mean for threshold.
            - tmp_folder (str): Temporary folder for parallel processing.
            - chunk_size (int, default=60000): Size of signal chunks for noise reduction.
            - padding (int, default=30000): Padding applied to each signal chunk.
            - n_fft (int, default=1024): Length of the windowed signal after padding for STFT.
            - win_length (int): Length of the window function for the STFT.
            - hop_length (int): Number of audio samples between adjacent STFT columns.
            - n_jobs (int, default=1): Number of parallel jobs to run.

        """
        self.sr = sr
        self.prop_decrease = prop_decrease
        self.verbose = verbose
        self.kwargs = kwargs

    def reduce_noise(self, audio_path):
        """
        Reduce noise from the given audio file.

        :param audio_path: Path to the audio file to be denoised.

        :returns: Denoised audio data.
        """
        y, sr = librosa.load(audio_path, sr=self.sr)
        y_denoised = nr.reduce_noise(
            y=y, sr=sr, prop_decrease=self.prop_decrease, **self.kwargs
        )

        if self.verbose:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(y, label="Noisy", color="grey")
            plt.plot(y_denoised, label="Denoised", color="blue")
            plt.legend()
            plt.title("Noisy vs. Denoised Signal")

            plt.subplot(2, 1, 2)
            plt.plot(y - y_denoised, color="red")
            plt.title("Difference (Residual Noise)")

            plt.tight_layout()
            plt.savefig("noisy_vs_denoised.png")

        return y_denoised


def extract_feature():
    pass


def vad():
    pass


def speaker_segmentation():
    pass


def speaker_embedding():
    pass


def speaker_clustering():
    pass


def diarize_conversation():
    pass
