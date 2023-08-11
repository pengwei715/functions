#TODO
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

def remove_noise_reverb():
    pass

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
