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
import os
import pathlib
import tempfile
from difflib import SequenceMatcher

import mlrun
import pytest
import whisper
from transcribe import Diarizator


expected_outputs = [
    "This is a speech to text test.",
    "In the heart of the stadium, "
    "cheers paint the air as the ball weaves its tale across the pitch. "
    "With each kick, players chase their dreams, guided by the rhythmic dance of teamwork. "
    "The crowd roars, a symphony of passion, "
    "as the game writes its unpredictable story on the field of destiny.",
]


@pytest.mark.parametrize("model_name", whisper.available_models()[:4])
@pytest.mark.parametrize("audio_path", ["./data", "./data/speech_01.mp3"])
def test_transcribe(model_name: str, audio_path: str):
    # Setting variables and importing function:
    artifact_path = tempfile.mkdtemp()
    transcribe_function = mlrun.import_function("function.yaml")
    temp_dir = tempfile.mkdtemp()

    # Running transcribe function:
    transcribe_run = transcribe_function.run(
        handler="transcribe",
        params={
            "input_path": audio_path,
            "model_name": model_name,
            "device": "cpu",
            "output_directory": temp_dir,
        },
        local=True,
        returns=["output_dir: path", "dataset: dataset", "errored_files"],
        artifact_path=artifact_path,
    )

    artifact_path += (
        f"/{transcribe_run.metadata.name}/{transcribe_run.metadata.iteration}/"
    )

    # Getting actual files from run (text and errored):
    input_files = (
        os.listdir(audio_path)
        if pathlib.Path(audio_path).is_dir()
        else [pathlib.Path(audio_path).name]
    )
    expected_text_files = sorted([f for f in input_files if f.endswith("mp3")])
    error_files = list(set(input_files) - set(expected_text_files))
    expected_text_files = [f.replace("mp3", "txt") for f in expected_text_files]
    text_files = sorted(os.listdir(temp_dir))

    # Check that the text files are saved in output_directory:
    assert text_files == expected_text_files

    # Check that the transcribed text was approximately (90%) generated from audio:
    for text_file, expected in zip(text_files, expected_outputs):
        with open(os.path.join(temp_dir, text_file), "r") as f:
            output = f.readlines()[0]
            ratio = SequenceMatcher(None, expected, output).ratio()
            assert ratio >= 0.9

    # Check that the dataframe is in the correct size:
    df = mlrun.get_dataitem(artifact_path + "dataset.parquet").as_df()
    assert len(df) == len(expected_text_files)

    # Check errored files:
    if isinstance(transcribe_run.outputs["errored_files"], str):
        actual_errored_files = []
    else:
        actual_errored_files = [
            os.path.basename(errored)
            for errored in transcribe_run.outputs["errored_files"].keys()
        ]
    assert actual_errored_files == error_files

    # Check output_dir:
    zip_dir = mlrun.get_dataitem(artifact_path + "output_dir.zip")
    assert zip_dir.kind == "file"


@pytest.mark.parametrize("audio_file", os.listdir("./data"))
def test_convert_to_support_format(audio_file):
    # Ensure the audio file is an MP3
    if not audio_file.endswith(".mp3"):
        pytest.skip("Skipping non-mp3 file")

    # Path to the audio file
    audio_path = os.path.join("./data", audio_file)

    # Create a diarizator object
    diarizator = Diarizator()

    # Convert to supported format
    converted_file_path = diarizator._convert_to_support_format(audio_path)

    # Check if the converted file exists
    assert os.path.exists(
        converted_file_path
    ), f"Converted file {converted_file_path} does not exist."

    # Check if the converted file is in the expected format (e.g., .wav)
    assert converted_file_path.endswith(
        ".wav"
    ), f"Converted file {converted_file_path} is not in .wav format."


@pytest.mark.parametrize("audio_file", os.listdir("./data"))
def test_split_audio_by_speaker(audio_file):
    # Ensure the audio file is an MP3
    if not audio_file.endswith(".mp3"):
        pytest.skip("Skipping non-mp3 file")

    # Path to the audio file
    audio_path = os.path.join("./data", audio_file)

    # Create a diarizator object
    diarizator = Diarizator()
    res, audio_file_path = diarizator._run(audio_path)
    
    segments = diarizator._split_audio(audio_file_path, res)
    assert len(segments) > 0, "No segments were found."


