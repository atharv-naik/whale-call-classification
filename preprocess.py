'''
preprocess.py
    Contains the pre-processing steps for the spectrogram images

Usage:
    python preprocess.py'''

import os
import aifc
from tqdm import tqdm
import numpy as np
import glob
from skimage.transform import resize
from matplotlib import mlab


"""
1. Pre Procesing:

We performed a pre-processing step to improve the horizontal (temporal dimension) and vertical (frequency dimension)
contrast of the spectrogram images which proved to be efficient in getting better results. Both kinds of contrast enhance-
ment were achieved by creating two moving average filters with only difference in their length (denoted filter A and filter B).

1.1 Temporal dimension:

Each row of the spectrogram was combined once with filter A and then with filter B. Since filter a had a much smaller length
compared to filter B the values of the output of the combination with filter A represented local averages for adjacent pixels
whereas the output of the combination with filter B represented global averages for neighborhoods of pixels. Contrast
enhancement was achieved by subtracting out these local averages from their corresponding rows, thereby emphasizing
more significant temporal changes.Contrast enhancement was performed by subtracting out the local averages from their
corresponding rows, thereby emphasizing more prominent temporal changes.

1.2 Frequency domain:

Each column of the spectrogram was combined sperately with filter A and filter B for contrast enhancement and to empha-
size more on significant differences in the frequency power distribution for every time step, the output of the combination
operation with filter A is subtracted from that of filter B, and the actual local averages subtracted from each corresponding
row.

Two new kinds of spectrogram images are produced from one original sample, thus doubling the number of samples in the
dataset.
"""


def ReadAIFF(file):
    ''' ReadAIFF Method
            Read AIFF and convert to numpy array

            Args: 
                file: string file to read 
            Returns:
                numpy array containing whale audio clip      

    '''
    s = aifc.open(file, 'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.frombuffer(strSig, np.short).byteswap()


def SpecGram(file, params=None):
    '''  The  function takes an audio file (specified as a string file), and creates a spectrogram representation of it.
         It  pre-process input for input shape uniformity 
         Args: 1. string file to read 
               2. The parameters for creating the spectrogram are passed as a dictionary in the "params" argument.  
         The output of the function is a pre-processed spectrogram matrix and arrays for the frequency and time bins, which are one-dimensional.

    '''
    s = ReadAIFF(file)
    # Convert to spectrogram
    P, freqs, bins = mlab.specgram(s, **params)
    m, n = P.shape
    # Ensure all image inputs to the CNN are the same size. If the number of time bins
    # is less than 59, pad with zeros
    if n < 59:
        Q = np.zeros((m, 59))
        Q[:, :n] = P
    else:
        Q = P
    return Q, freqs, bins


def slidingWindowV(P, inner=3, outer=64, maxM=50, minM=7, maxT=59, norm=True):
    ''' Enhance the contrast along frequency dimension '''

    Q = P.copy()
    m, n = Q.shape
    if norm:
        # segment to remove  extreme values
        mval, sval = np.mean(Q[minM:maxM, :maxT]), np.std(Q[minM:maxM, :maxT])
        fact_ = 1.5
        Q[Q > mval + fact_*sval] = mval + fact_*sval
        Q[Q < mval - fact_*sval] = mval - fact_*sval
        Q[:minM, :] = mval
    # Setting up the local mean window
    wInner = np.ones(inner)
    # Setting up the overall mean window
    wOuter = np.ones(outer)
    # Removing  overall mean and local mean using np.convolve

    for i in range(maxT):
        Q[:, i] = Q[:, i] - (np.convolve(Q[:, i], wOuter, 'same') -
                             np.convolve(Q[:, i], wInner, 'same'))/(outer - inner)
    Q[Q < 0] = 0.
    return Q[:maxM, :]


def slidingWindowH(P, inner=3, outer=32, maxM=50, minM=7, maxT=59, norm=True):
    ''' Enhance the contrast along temporal dimension '''
    Q = P.copy()
    m, n = Q.shape
    if outer > maxT:
        outer = maxT
    if norm:
        # Cutting off extreme values
        mval, sval = np.mean(Q[minM:maxM, :maxT]), np.std(Q[minM:maxM, :maxT])
        fact_ = 1.5
        Q[Q > mval + fact_*sval] = mval + fact_*sval
        Q[Q < mval - fact_*sval] = mval - fact_*sval
        Q[:minM, :] = mval
    # Setting up the local mean window and overall mean window
    wInner = np.ones(inner)
    wOuter = np.ones(outer)
    if inner > maxT:
        return Q[:maxM, :]
    # Removing overall mean and local mean using np.convolve

    for i in range(maxM):
        Q[i, :maxT] = Q[i, :maxT] - (np.convolve(Q[i, :maxT], wOuter, 'same') -
                                     np.convolve(Q[i, :maxT], wInner, 'same'))/(outer - inner)
    Q[Q < 0] = 0.
    return Q[:maxM, :]


def get_file_id(file):
    id, extension = os.path.splitext(file)
    return id


def extract_labels(file):
    name, extension = os.path.splitext(file)
    label = name[-1]
    return int(label)


def extract_featuresV(file, params=None):
    '''The function is used for obtaining a spectrogram representation of an audio file with vertically-enhanced contrast.
       suitable for input into a Convolutional Neural Network (CNN).
       The audio file is specified as a string in the "file" argument,
       and the spectrogram parameters are passed in as a dictionary in the "params" argument. 
       The output of the function is a 2-dimensional numpy array, which is an image with vertically-enhanced contrast.    

    '''
    P, freqs, bins = SpecGram(file, params)
    Q = slidingWindowV(P, inner=3, maxM=50, maxT=bins.size)
    # Resize spectrogram image into a square matrix
    Q = resize(Q, (64, 64), mode='edge')
    return Q


def extract_featuresH(file, params=None):
    ''' The function is used  for obtaining a spectrogram of an audio file with an emphasis on horizontal contrast, 
       suitable for input into a Convolutional Neural Network (CNN). 
       The audio file is specified as a string in the "file" argument, and 
       the parameters for creating the spectrogram are passed in as a dictionary in the "params" argument. 
       The result of the function is a 2-dimensional numpy array, which is an image with horizontally-enhanced contrast. 


    '''
    P, freqs, bins = SpecGram(file, params)
    W = slidingWindowH(P, inner=3, outer=32, maxM=50, maxT=bins.size)
    # Resize spectrogram image into a square matrix
    W = resize(W, (64, 64), mode='edge')
    return W


def get_training_data():
    ''' method of obtaining data'''

    # Spectrogram parameters
    params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
    # Load in the audio files from the training dataset
    path = 'aiff/train'
    filenames = glob.glob(path+'/[01]/*.aiff')
    """
     For each audio file, we obation  the spectrograms with vertically-enhanced contrast 
     and the spectrograms with horizontally-enhanced contrast. This in 
     effect doubles the amount of data for training, and presents the CNN with different 
     perspectives of the same spectrogram image of the original audio file 

    """

    print('Extracting Training Features')
    training_featuresV = np.array([extract_featuresV(
        x, params=params) for x in tqdm(filenames, desc="Extracting test features V")])
    training_featuresH = np.array([extract_featuresH(
        x, params=params) for x in tqdm(filenames, desc="Extracting test features H")])
    # Concatenate the two feature matrices together to form a double-length feature matrix
    X_train = np.append(training_featuresV, training_featuresH, axis=0)
    # Axis 0 indicates the number of examples, Axis 1 and 2 are the features (64x64 image
    # spectrograms). Add Axis 3 to indicate 1 channel (depth of 1 for spectrogram image) for
    # compatibility with Keras CNN model
    X_train = X_train[:, :, :, np.newaxis]

    # Extract labels for the training dataset. Since the vertically-enhanced and
    # horizontally-enhanced images are concatenated to form a training dataset twice as long,
    # append a duplicate copy of the training labels to form a training label vector
    # twice as long

    print('Extracting Training Labels')
    Y_train = np.array([extract_labels(x) for x in tqdm(
        filenames, desc="Extracting training labels")])
    Y_train = np.append(Y_train, Y_train)

    # Do not append a duplicate copy of the test labels to form a test label vector twice
    # as long, since the two feature matrices were not concatenated together previously.
    # The number of elements in the test label vector is the number of original audio
    # files in the test dataset

    return X_train, Y_train


def get_test_data():

    # Spectrogram parameters
    params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
    # Load in the audio files from the test dataset
    path = 'aiff/test'
    filenames = glob.glob(path+'/*.aiff')

    '''
    For each audio file, extract the spectrograms with vertically-enhanced contrast separately from the spectrograms with horizontally-enhanced contrast. This in effect doubles the amount of data for training, and presents the CNN with different perspectives of the same spectrogram image of the original audio file 
    '''

    print('Extracting File id')
    file_id = np.array([get_file_id(x)
                       for x in tqdm(filenames, desc="Extracting File id")])
    file_id = np.append(file_id, file_id)

    print('Extracting Test Features')
    X_testV = np.array([extract_featuresV(x, params=params)
                       for x in tqdm(filenames, desc="Extracting test features V")])
    X_testH = np.array([extract_featuresH(x, params=params)
                       for x in tqdm(filenames, desc="Extracting test features H")])
    X_testV = X_testV[:, :, :, np.newaxis]
    X_testH = X_testH[:, :, :, np.newaxis]
    X_test = np.append(X_testV, X_testV, axis=0)
    X_test = X_test[:, :, :, np.newaxis]

    return file_id, X_test, X_testV, X_testH


def main():
    # preprocess and save the training and test data
    X_train, Y_train = get_training_data()
    file_id, X_test, X_testV, X_testH = get_test_data()
    np.savez('training_data_preprocessed.npz',
             X_train=X_train, Y_train=Y_train)
    np.savez('test_data_preprocessed.npz', file_id=file_id,
             X_test=X_test, X_test_V=X_testV, X_test_H=X_testH)


if __name__ == '__main__':
    main()
