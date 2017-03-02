import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

class Dataset:


    def __init__(self, path,f=0, normalize='y'): # normalize is set by default to 'y' or yes, meaning that it will automatically normalize the data.

        ''' The file has the following structure:

                    {
                    <audio_path (str)> : list of (<label (int)>, <chirplets (numpy array))>,
                    <audio_path (str)> : list of (<label (int)>, <chirplets (numpy array))>,
                    <audio_path (str)> : list of (<label (int)>, <chirplets (numpy array))>,
                    …
                    }
                    Example of an element: 'test/dr7/mkjl0/sx20.wav': [(5, array([[ 0.1096134, 0.50897387,..]])),(6, array([[ 0.21221525, 0.23542235,..]])),...]
                     dr7 -> Dialect, mkjl0 -> Speaker, sx20.wav -> audio file
                     -> Each data associated with the label is a 3x64 dimensional numpy array
                     3 frames each of 20 milliseconds and 64 features(frequency bands)
        '''

        self.__path = None  # The path of the .pkl file containing the data
        self.__data = []  # To store the data
        self.__labels = []  # To store the labels of the corresponding data
        self.__speakers = []  # To store the speakers corresponding to the data
        self.__dialects = []  # To store the dialects corresponding to the data

        self.__path = path
        dataset = joblib.load(self.__path)   # joblib.load for loading the file(which contains a dictionary of filenames as keys and datavalues as values) to the dataset variable
        dataset = OrderedDict(sorted(dataset.items()))


        if(f==0):
            for (k, v) in dataset.items():
                d = k.split("/")[1]                  # Get dialect of the current audio from the key
                s = k.split("/")[2]                  # Get speaker of the current audio from the key
                #self.__data += [i[1][0] for i in v]  # Get the list of the chirplet data for the current audio and append it to the __data array
                self.__data += [i[1] for i in v]     # Get the list of the chirplet data for the current audio and append it to the __data array
                label = [i[0] for i in v]            # Get the list of labels of the data for the current audio
                self.__labels += label               # Append the label list to the __labels array
                self.__speakers += [s] * len(label)  # Add the same speaker in the speakers array for the current audio(since any single audio is only spoken by one speaker)
                self.__dialects += [d] * len(label)  # Add the same speaker in the speakers array for the current audio
        else:
            for (k, v) in dataset.items():
                d = k.split("/")[1]                  # Get dialect of the current audio from the key
                s = k.split("/")[2]                  # Get speaker of the current audio from the key
                self.__data += [i[1][0] for i in v]  # Get the list of the chirplet data for the current audio and append it to the __data array
                #self.__data += [i[1] for i in v]     # Get the list of the chirplet data for the current audio and append it to the __data array
                label = [i[0] for i in v]            # Get the list of labels of the data for the current audio
                self.__labels += label               # Append the label list to the __labels array
                self.__speakers += [s] * len(label)  # Add the same speaker in the speakers array for the current audio(since any single audio is only spoken by one speaker)
                self.__dialects += [d] * len(label)  # Add the same speaker in the speakers array for the current audio

        if(normalize=='y'):                      # If normalize is set to yes ,then use Standard scaling and also reshape the data(if not already in proper shape)
            self.Scale_reshape()
        else:                                    # Else, just reshape the data if required
            self.reshape_it()

    def Scale_reshape(self):                                           # Function to Standard Scale(Mean:0 and Variance:1) and reshape the data
        scaler = StandardScaler()                                      # Using StandardScaler class for Standard Scaling.
        if(self.__data[0].ndim==1):                                    # For t-SNE , input data should be of one dimensional shape
            tmp = self.__data                                          # Temporary array to concatenate the data matrix so that all the data is arranged according to the 64 features of the data.
            scaler.fit(tmp)                                            # Fitting the tmp array for Standard Scaling
            self.__data = scaler.transform(self.__data)                # Transform each data element according to the scaler trained.
        else:
            tmp = np.concatenate(self.__data)
            scaler.fit(tmp)
            x, y = self.__data[0].shape                                # If data is not in 1D shape, then take its shape dimensions
            for i in range(len(self.__data)):
                self.__data[i] = scaler.transform(self.__data[i])
                self.__data[i] = np.reshape(self.__data[i], (x*y))     # After Scaling, reshape the data to 1D shape

    def reshape_it(self):                                              # Function to reshape the data when no normalization is requested.
        if (self.__data[0].ndim != 1):
            x, y = self.__data[0].shape
            for i in range(len(self.__data)):
                self.__data[i] = np.reshape(self.__data[i], (x * y))

# Return the elements as numpy array.
    @property
    def data(self):
        return np.asarray(self.__data)

    @property
    def labels(self):
        return np.asarray(self.__labels)

    @property
    def speakers(self):
        return np.asarray(self.__speakers)

    @property
    def dialects(self):
        return np.asarray(self.__dialects)