# Whale Call Classification

This project aims to classify Blue whale calls into A-calls or not based on a training dataset provided as a part of ISI DataFest Integration 2023. A-Calls are the most commonly heard vocalization of Blue whales and play an important role in their communication and behavior.

The project uses a Convolutional Neural Network (CNN) to classify the Blue whale calls. The CNN model is trained on a training dataset consisting of 1671 samples and validated on a validation dataset consisting of 375 samples. The trained model is then tested on a separate testing dataset consisting of about 26000 samples.

## Installation

Clone the repo and cd into it. To install the required packages, run the following command:
```
pip install -r requirements.txt
```
Download the dataset from the competetion link [here](https://www.kaggle.com/competitions/datafestintegration2023/data) or using the kaggle API, run:
```
kaggle competitions download -c datafestintegration2023
```
Make sure the dataset is in the same directory as the working directory.

## Usage

First preprocess the data. Install ffmpeg which is a dependancy for ```make_aiff.py``` and then preprocess the data. Run the following commands:
```
sudo apt install ffmpeg
python make_aiff.py
python preprocess.py
```

To train the model, run the following command:

```
python train.py
```

## Dataset

The dataset consists of labeled audio files of Blue whale calls in WAV format. The files are categorized as A-call or not based on their spectrogram.

## Model Architecture

The CNN model consists of 3 convolutional layers, followed by 2 fully connected layers, and an output layer. The model is trained with binary cross-entropy loss and Adam optimizer.

## Results

The model achieved an accuracy of 97% on the test set.

## Authors

- Arghya Maity (am20ms038@iiserkol.ac.in)
- Atharv Naik (naa20ms195@iiserkol.ac.in)
- Somnath Roy (sr20ms063@iiserkol.ac.in)

## License

This project is licensed under the [MIT](LICENSE) license.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Arg-10/Whale-Call-Classification/blob/main/LICENSE)
