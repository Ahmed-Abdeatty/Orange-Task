# Arda Mavi
# some of this code is taken from
# https://gist.github.com/ardamavi/a7d06ff8a315308771c70006cf494d69
import os
import numpy as np
from os import listdir
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

class get_dataset:
    def __init__(self):
        # Settings:
        self.img_size = 64
        self.grayscale_images = True
        self.num_class = 10
        self.test_size = 0.2

    def get_img(self,data_path):
        # Getting image array from path:
        img = imread(data_path, flatten=self.grayscale_images)
        img = imresize(img, (self.img_size, self.img_size, 1 if self.grayscale_images else 3))
        return img

    def data_preproc(self,dataset_path='Dataset',use_pca=0):
        """
        input:
            use_pca: int variable to indicate wether to use PCA (> 0) or not (<= 0)
        output: 2 object contains data splitted into test, train, and validation 
                1 object has splitting suitable for CNN and the other suitable for
                SVM like classifiers
        """
        # Getting all data from data path:
        labels = listdir(dataset_path) # Geting labels
        Y = []
        X = []
        X1 = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = self.get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(i)
        
        # Create dateset:
        """
        the data right now is in an int8 format, so before you feed it into the network you 
        need to convert its type to float32, and you also have to rescale the pixel values 
        in range 0 - 1 inclusive. 
        """
        X = 1-np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        # obtain 20% of the data as testing data
        X, X_test, Y_svm, Y_test_svm = train_test_split(X, Y, test_size=self.test_size, random_state=42)
        # split the remaining 80% into 80% training and 20% validation
        X_train, X_valid, Y_train_svm, Y_valid_svm = train_test_split(X, Y_svm, test_size=self.test_size, random_state=42)
        
        # convert the data into formate suitable for classifiers like SVM
        X_test_svm = self.flatten_img(X_test)
        X_train_svm = self.flatten_img(X_train)
        X_valid_svm = self.flatten_img(X_valid)

        if use_pca > 0:
            pca = TruncatedSVD(n_components=use_pca)
            X_train_svm = pca.fit_transform(X_train_svm)
            X_valid_svm = pca.transform(X_valid_svm)
            X_test_svm = pca.transform(X_test_svm)
        # convert the data into formate suitable for CNN
        X_train = X_train.reshape(-1, 64, 64, 1)
        X_valid = X_valid.reshape(-1, 64, 64, 1)
        X_test = X_test.reshape(-1, 64, 64, 1)
        
        # convert the class labels into a one-hot encoding vector to be suitable for CNN representation 
        Y_train = to_categorical(Y_train_svm, self.num_class)
        Y_test = to_categorical(Y_test_svm, self.num_class)
        Y_valid = to_categorical(Y_valid_svm, self.num_class)
        
        # return data as objects
        cnn_ds = data_split()
        cnn_ds.Y_train = Y_train
        cnn_ds.Y_test = Y_test
        cnn_ds.Y_valid = Y_valid
        cnn_ds.X_train = X_train
        cnn_ds.X_test = X_test
        cnn_ds.X_valid = X_valid

        svm_ds = data_split()
        svm_ds.Y_train = Y_train_svm
        svm_ds.Y_test = Y_test_svm
        svm_ds.Y_valid = Y_valid_svm
        svm_ds.X_train = X_train_svm
        svm_ds.X_test = X_test_svm
        svm_ds.X_valid = X_valid_svm
        return cnn_ds, svm_ds

    # flatten a set of 28X28 images into a set of 4096 feature vectors
    def flatten_img(self,X):
        """
        input:
            X: set of 28X28 images 
        output:
            set of 4096 feature vectors
        """
        flat_X = []
        for i in range(X.shape[0]):
            temp = X[i]
            flat_X.append(temp.reshape(4096))
        return flat_X
class data_split:
    def __init__(self):
        self.Y_train = None
        self.Y_test = None
        self.Y_valid = None
        self.X_train = None
        self.X_test = None
        self.X_valid = None


    