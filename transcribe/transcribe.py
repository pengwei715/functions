# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pathlib
import tempfile
import librosa
import mlrun
import pandas as pd
import whisper
from tqdm.auto import tqdm
import json
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple


def transcribe(
    context: mlrun.MLClientCtx,
    input_path: str,
    model_name: str = "base",
    device: Literal["cuda", "cpu"] = None,
    decoding_options: dict = None,
    output_directory: str = None,
) -> Tuple[pathlib.Path, pd.DataFrame, dict]:
    """
    Transcribe audio files into text files and collect additional data.
    The end result is a directory of transcribed text files
     and a dataframe containing the following columns:

    * audio_file - The original audio file name.
    * transcription_file - The transcribed text file name in the output directory.
    * language - The detected language in the audio file.
    * length - The length of the audio file.
    * rate_of_speech - The number of words divided by the audio file length.

    :param context:               MLRun context.
    :param input_path:            A directory of the audio files or a single file to transcribe.
    :param output_directory:      Path to a directory to save all transcribed audio files.
    :param model_name:            One of the official model names listed by `whisper.available_models()`.
    :param device:                Device to load the model. Can be one of {"cuda", "cpu"}.
                                  Default will prefer "cuda" if available.
    :param decoding_options:      A dictionary of options to construct a `whisper.DecodingOptions`.

    :returns: A tuple of:

              * Path to the output directory.
              * A dataframe dataset of the transcribed file names.
              * A dictionary of errored files that were not transcribed.
    """
    # Set output directory:
    if output_directory is None:
        output_directory = tempfile.mkdtemp()

    # Load the model:
    context.logger.info(f"Loading whisper model: '{model_name}'")
    model = whisper.load_model(model_name, device=device)
    context.logger.info("Model loaded.")

    # Prepare the dataframe and errors to be returned:
    df = pd.DataFrame(
        columns=[
            "audio_file",
            "transcription_file",
            "language",
            "length",
            "rate_of_speech",
        ]
    )
    errors = {}

    # Create the output directory:
    output_directory = pathlib.Path(output_directory)
    if not output_directory.exists():
        output_directory.mkdir()

    # Go over the audio files and transcribe:
    audio_files_path = pathlib.Path(input_path).absolute()
    is_dir = True
    if audio_files_path.is_dir():
        audio_files = list(audio_files_path.rglob("*.*"))
    elif audio_files_path.is_file():
        is_dir = False
        audio_files = [audio_files_path]
    else:
        raise ValueError(
            f"audio_files {str(audio_files_path)} must be either a directory path or a file path"
        )

    for i, audio_file in enumerate(tqdm(audio_files, desc="Transcribing", unit="file")):
        try:
            transcription, length, rate_of_speech, language = _single_transcribe(
                audio_file=audio_file,
                model=model,
                decoding_options=decoding_options,
            )

        except Exception as exception:
            # Collect the exception:
            context.logger.warn(f"Error in file: '{audio_file}'")
            errors[str(audio_file)] = str(exception)
        else:
            # Write the transcription to file:
            saved_filename = (
                str(audio_file.relative_to(audio_files_path)).split(".")[0]
                if is_dir
                else audio_file.stem
            )
            transcription_file = output_directory / f"{saved_filename}.txt"
            transcription_file.parent.mkdir(exist_ok=True, parents=True)
            with open(transcription_file, "w") as fp:
                fp.write(transcription)

            # Note in the dataframe:
            df.loc[i - len(errors)] = [
                str(audio_file.relative_to(audio_files_path)),
                str(transcription_file.relative_to(output_directory)),
                language,
                length,
                rate_of_speech,
            ]
    # Return the dataframe:
    context.logger.info(f"Done:\n{df.head()}")

    return output_directory, df, errors


def _single_transcribe(
    audio_file: pathlib.Path,
    model: whisper.Whisper,
    decoding_options: dict = None,
) -> Tuple[str, int, float, str]:
    decoding_options = decoding_options or dict()
    # Load the audio:
    audio = whisper.audio.load_audio(file=str(audio_file))
    # Get audio length:
    length = librosa.get_duration(path=audio_file)
    # Transcribe:
    result = model.transcribe(audio=audio, **decoding_options)
    # Unpack the model's result:
    transcription = result["text"]
    language = result.get("language") or decoding_options.get("language", "")

    # Calculate rate of speech (number of words / audio length):
    rate_of_speech = len(transcription.split()) / length

    return transcription, length, rate_of_speech, language


#Using Nemo to do speaker diarization we need the following steps:
#1. Voice Activity Detection, which part of the audio is speech and which part is not
#2. Speaker Embedding is used to extract the features of the speech
#3. Clustering the embedding get the clusters of speakers
#4. Multiscale Diarization decoder is used to obtain the speaker profile and estimated number of speakers


# Parameters for VAD (Voice Activity Detection)
@dataclass
class VADParameters:
    window_length_in_sec: float = 0.15  # Window length in seconds for VAD context input
    shift_length_in_sec: float = (
        0.01  # Shift length in seconds to generate frame-level VAD prediction
    )
    smoothing: str = "median"  # Type of smoothing method (e.g., "median")
    overlap: float = 0.5  # Overlap ratio for overlapped mean/median smoothing filter
    onset: float = 0.1  # Onset threshold for detecting the beginning of speech
    offset: float = 0.1  # Offset threshold for detecting the end of speech
    pad_onset: float = 0.1  # Additional duration before each speech segment
    pad_offset: float = 0  # Additional duration after each speech segment
    min_duration_on: float = 0  # Threshold for small non-speech deletion
    min_duration_off: float = 0.2  # Threshold for short speech segment deletion
    filter_speech_first: bool = True  # Whether to apply a speech-first filter


# Main VAD Configuration
@dataclass
class VADConfig:
    model_path: str = "vad_multilingual_marblenet"  # Path to the VAD model
    external_vad_manifest: Optional[
        str
    ] = None  # Optional path to an external VAD manifest
    parameters: VADParameters = field(
        default_factory=VADParameters
    )  # Nested VAD parameters


# Parameters for Speaker Embeddings
@dataclass
class SpeakerEmbeddingsParameters:
    window_length_in_sec: List[float] = field(
        default_factory=lambda: [1.5, 1.25, 1.0, 0.75, 0.5]
    )  # Window lengths for speaker embeddings
    shift_length_in_sec: List[float] = field(
        default_factory=lambda: [0.75, 0.625, 0.5, 0.375, 0.25]
    )  # Shift lengths for speaker embeddings
    multiscale_weights: List[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 1]
    )  # Weights for each scale
    save_embeddings: bool = True  # Whether to save the speaker embeddings


# Main Speaker Embeddings Configuration
@dataclass
class SpeakerEmbeddingsConfig:
    model_path: str = "nvidia/speakerverification_en_titanet_large"  # Path to the speaker embeddings model
    parameters: SpeakerEmbeddingsParameters = field(
        default_factory=SpeakerEmbeddingsParameters
    )  # Nested speaker embeddings parameters


# Parameters for Clustering
@dataclass
class ClusteringParameters:
    oracle_num_speakers: bool = False  # Whether to use the oracle number of speakers
    max_num_speakers: int = 8  # Maximum number of speakers per recording
    enhanced_count_thres: int = 80  # Threshold for enhanced speaker counting
    max_rp_threshold: float = 0.25  # Max range of p-value search for resegmentation
    sparse_search_volume: int = 30  # Number of values to examine for sparse search
    maj_vote_spk_count: bool = (
        False  # Whether to take a majority vote for speaker count
    )


# Main Clustering Configuration
@dataclass
class ClusteringConfig:
    parameters: ClusteringParameters = field(
        default_factory=ClusteringParameters
    )  # Nested clustering parameters


# Parameters for MSDD (Multiscale Diarization Decoder) Model
@dataclass
class MSDDParameters:
    use_speaker_model_from_ckpt: bool = (
        True  # Whether to use the speaker model from the checkpoint
    )
    infer_batch_size: int = 25  # Batch size for MSDD inference
    sigmoid_threshold: List[float] = field(
        default_factory=lambda: [0.7]
    )  # Sigmoid threshold for binarized speaker labels
    seq_eval_mode: bool = (
        False  # Whether to use oracle number of speakers for sequence evaluation
    )
    split_infer: bool = True  # Whether to split the input audio for inference
    diar_window_length: int = (
        50  # Length of the split short sequence when split_infer is True
    )
    overlap_infer_spk_limit: int = (
        5  # Limit for estimated number of speakers for overlap inference
    )


# Main MSDD Configuration
@dataclass
class MSDDConfig:
    model_path: str = "diar_msdd_telephonic"  # Path to the MSDD model
    parameters: MSDDParameters = field(
        default_factory=MSDDParameters
    )  # Nested MSDD parameters


# Main Diarization Configuration
@dataclass
class DiarizationConfig:
    manifest_filepath: str  # Path to the manifest file
    out_dir: str  # Output directory
    oracle_vad: bool = False  # Whether to use oracle VAD
    collar: float = 0.25  # Collar value for scoring
    ignore_overlap: bool = True  # Whether to ignore overlap segments
    vad: VADConfig = field(default_factory=VADConfig)  # Nested VAD configuration
    speaker_embeddings: SpeakerEmbeddingsConfig = field(
        default_factory=SpeakerEmbeddingsConfig
    )  # Nested speaker embeddings configuration
    clustering: ClusteringConfig = field(
        default_factory=ClusteringConfig
    )  # Nested clustering configuration
    msdd_model: MSDDConfig = field(
        default_factory=MSDDConfig
    )  # Nested MSDD configuration


# Form the speaker diarization pipeline and get the RTTM result
def _diarizate(audio_path: str, manifest_filepath: str, **kwags: dict):-> str:
    pass


