'''
make_aiff.py
    Converts all .wav files in the train and test directories to .aiff files.

Usage:
    python make_aiff.py'''

import os
from tqdm import tqdm
from pydub import AudioSegment


# convert all wav files to aiff files using audio segment package

AudioSegment.converter = "/usr/bin/ffmpeg"  # using ffmpeg package as a dependency for audio segment


def convert_to_aiff(file_path, new_file_path):
    # Load the .wav file
    audio = AudioSegment.from_wav(file_path)
    # Save the .aiff file
    audio.export(new_file_path, format="aiff")


def main():
    # Converting training files
    # List the directories containing the .wav files
    labels = ["1", "0"]
    to_directory = "aiff/train/"
    directories = ["train/train/0", "train/train/1"]

    # Get a list of all the .wav files in the directory
    for directory in directories:
        label = directory[-1]
        wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
        # Loop through each .wav file
        for wav_file in tqdm(wav_files, desc=directory):
            # Get the file path of the .wav file
            file_path = os.path.join(directory, wav_file)
            to_file_path = os.path.join(to_directory, wav_file)
            # Get the new file path for the .aiff file
            new_file_path = os.path.splitext(to_file_path)[0] + label + ".aiff"
            # Convert the .wav file to .aiff format
            convert_to_aiff(file_path, new_file_path)


    # Converting test files
    to_test_directory = "aiff/test/"
    from_test_directory = "test/test"
    # Get a list of all the .wav files in the from_test_directory
    wav_files = [f for f in os.listdir(from_test_directory) if f.endswith(".wav")]
    # Loop through each .wav file
    for wav_file in tqdm(wav_files, desc=from_test_directory):
        # Get the file path of the .wav file
        file_path = os.path.join(from_test_directory, wav_file)
        to_file_path = os.path.join(to_test_directory, wav_file)
        # Get the new file path for the .aiff file
        new_file_path = os.path.splitext(to_file_path)[0] + ".aiff"
        # Convert the .wav file to .aiff format
        convert_to_aiff(file_path, new_file_path)


if __name__ == "__main__":
    main()
