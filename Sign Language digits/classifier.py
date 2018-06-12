from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
##########################################################################
#Written By: Ahmed Abdelatty email: ahmed.abdelatty@utdallas.edu 9/6/2018
##########################################################################
class classifier:
    

    def CNN(self,ds,batch_size=64,epochs=65,num_classes=10,dropout=True,drop_perc = [.25,.25,.35,.4,.5]):
        """
        input:
            ds: data_split object
            drop_perc: vector of lenght 5 defining the percent of dropout at every layer
        output:
            model accuracy on training, validation, testing data
        """
        """
        You'll use 4 convolutional layers:

        The first layer will have 32-3 x 3 filters,
        The second layer will have 64-3 x 3 filters and
        The third layer will have 128-3 x 3 filters.
        The fourth layer will have 128-3 x 3 filters
        In addition, there are four max-pooling layers each of size 2 x 2.
        """
        
        cnn = Sequential()
        cnn.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(64,64,1),padding='same'))
        cnn.add(MaxPooling2D((2, 2),padding='same'))
        if dropout:
            cnn.add(Dropout(drop_perc[0])) 
        cnn.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        cnn.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        if dropout:
            cnn.add(Dropout(drop_perc[1])) 
        cnn.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
        cnn.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        if dropout:
            cnn.add(Dropout(drop_perc[2]))
        cnn.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        cnn.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        if dropout:
            cnn.add(Dropout(drop_perc[3])) 
        cnn.add(Flatten())
        cnn.add(Dense(256, activation='relu'))
        if dropout:
            cnn.add(Dropout(drop_perc[4])) 
                         
        cnn.add(Dense(num_classes, activation='softmax'))

        cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        cnn.summary()
        fashion_train = cnn.fit(ds.X_train, ds.Y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(ds.X_valid, ds.Y_valid))
        train_eval = cnn.evaluate(ds.X_train, ds.Y_train, verbose=0)
        valid_eval = cnn.evaluate(ds.X_valid, ds.Y_valid, verbose=0)
        test_eval = cnn.evaluate(ds.X_test, ds.Y_test, verbose=0)
        return train_eval[1], valid_eval[1],test_eval[1]

    def NB(self,ds,alphas = [.5,1,1.5,2,2.5,3,5,7]):
        """
        input:
            ds: data_split object
            neighbors: tuning parameter (N)
        output:
            accuracy
        """
        best = 0
        best_model = None
        for alpha in alphas:
            text_clf = MultinomialNB(fit_prior=False,alpha=alpha)
            text_clf.fit(ds.X_train, ds.Y_train)
            predicted = text_clf.predict(ds.X_valid)
            temp = np.mean(predicted == ds.Y_valid)
            if temp > best:
                best_model = text_clf
                best = temp
        predicted = best_model.predict(ds.X_test)
        return np.mean(predicted == ds.Y_test)
    def KNN(self,ds,neighbors = [1,3,5,7,10,15]):
        """
        input:
            ds: data_split object
            neighbors: tuning parameter (N)
        output:
            accuracy
        """
        best = 0
        best_model = None
        for k in neighbors:
            text_clf = KNeighborsClassifier(n_neighbors=k)
            text_clf.fit(ds.X_train, ds.Y_train)
            predicted = text_clf.predict(ds.X_valid)
            temp = np.mean(predicted == ds.Y_valid)
            if temp > best:
                best_model = text_clf
                best = temp
        predicted = best_model.predict(ds.X_test)
        return np.mean(predicted == ds.Y_test)

    def SVM(self,ds,slack = [1,.1,.2,.25,2,5,10,20]):
        """
        input:
            ds: data_split object
            slach: tuning parameter (slack constant C)
        output:
            accuracy
        """
        best = 0
        best_model = None
        for c in slack:
            text_clf = svm.LinearSVC(C=c)
            text_clf.fit(ds.X_train, ds.Y_train)
            predicted = text_clf.predict(ds.X_valid)
            temp = np.mean(predicted == ds.Y_valid)
            if temp > best:
                best_model = text_clf
                best = temp
        predicted = best_model.predict(ds.X_test)
        return np.mean(predicted == ds.Y_test)

    def AdaBoost(self,ds,estimators = [50,100,200]):
        """
        input:
            ds: data_split object
            estimators: tuning parameter (number of models used in boosting)
        output:
            accuracy
        """
        best = 0
        best_model = None
        for n in estimators:
            text_clf = AdaBoostClassifier(n_estimators=n)
            text_clf.fit(ds.X_train, ds.Y_train)
            predicted = text_clf.predict(ds.X_valid)
            temp = np.mean(predicted == ds.Y_valid)
            if temp > best:
                best_model = text_clf
                best = temp
        predicted = best_model.predict(ds.X_test)
        return np.mean(predicted == ds.Y_test)
    

    def GradientBoosting(self,ds,estimators = [50,100,200]):
        """
        input:
            ds: data_split object
            estimators: tuning parameter (number of models used in boosting)
        output:
            accuracy
        """
        best = 0
        best_model = None
        for n in estimators:
            text_clf = GradientBoostingClassifier(n_estimators=n)
            text_clf.fit(ds.X_train, ds.Y_train)
            predicted = text_clf.predict(ds.X_valid)
            temp = np.mean(predicted == ds.Y_valid)
            if temp > best:
                best_model = text_clf
                best = temp
        predicted = best_model.predict(ds.X_test)
        return np.mean(predicted == ds.Y_test)

    # call all 5 classification techniques on our data and print the accuracy
    def classify(self,ds,neighbors = [1,3,5,7,10,15]
        ,slack = [1,.1,.2,.25,2,5,10,20],estimators = [50,100,200]):
        """
        input:
            ds: data_split object
            neighbors: KNN tuning parameter
            slack: SVM tuning parameter
            estimators: GradientBoosting, AdaBoost tuning parameter
        output:
            classification results
        """
        
        SVM_result = self.SVM(ds,slack)
        print("SVM    " + str(SVM_result))
        GB_result = self.GradientBoosting(ds,estimators)
        print("GradientBoosting    " + str(GB_result))
        AB_result = self.AdaBoost(ds,estimators)
        print("AdaBoost    " + str(AB_result))
        KNN_result = self.KNN(ds,neighbors)
        print("KNN    " + str(KNN_result))

        return SVM_result, GB_result, AB_result, KNN_result
        
        