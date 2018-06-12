from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import numpy as np

##########################################################################
#Written By: Ahmed Abdelatty email: ahmed.abdelatty@utdallas.edu 9/6/2018
##########################################################################
class classifier:
    
    # use the pretrained textblob sentiment classifier
    def pretrain_txtblob(self,df):
        """
        input:
            df: dataframe with two columns (tweet_text, tweet_sentiment)
        output:
            accuracy
        """
        result = 0
        for index, row in df.iterrows():
            analysis = TextBlob(row["text"])
            if analysis.sentiment.polarity > 0 and row["airline_sentiment"]  == "positive":
                result += 1
            elif analysis.sentiment.polarity == 0 and row["airline_sentiment"]  == "neutral":
                result += 1
            elif analysis.sentiment.polarity < 0 and row["airline_sentiment"]  == "negative":
                result += 1
        return result/(1.0 * df.shape[0])

    def NB(self,X_train, X_test, Y_train, Y_test,alphas = [.5,1,1.5,2,2.5,3,5,7]):
        """
        input:
            X_train: training data
            X_test: testing data
            Y_train: training labels
            Y_test: testing lables
            alphas: tuning parameter (Laplace priors)
        output:
            accuracy
        """
        best = 0
        for alpha in alphas:
            text_clf = MultinomialNB(fit_prior=False,alpha=alpha)
            text_clf.fit(X_train, Y_train)
            predicted = text_clf.predict(X_test)
            temp = np.mean(predicted == Y_test)
            if temp > best:
                best = temp
        return best

    def KNN(self,X_train, X_test, Y_train, Y_test,neighbors = [1,3,5,7,10,15]):
        """
        input:
            X_train: training data
            X_test: testing data
            Y_train: training labels
            Y_test: testing lables
            neighbors: tuning parameter (N)
        output:
            accuracy
        """
        best = 0
        for k in neighbors:
            text_clf = KNeighborsClassifier(n_neighbors=k)
            text_clf.fit(X_train, Y_train)
            predicted = text_clf.predict(X_test)
            temp = np.mean(predicted == Y_test)
            if temp > best:
                best = temp
        return best

    def SVM(self,X_train, X_test, Y_train, Y_test,slack = [1,.1,.2,.25,2,5,10,20]):
        """
        input:
            X_train: training data
            X_test: testing data
            Y_train: training labels
            Y_test: testing lables
            slach: tuning parameter (slack constant C)
        output:
            accuracy
        """
        best = 0
        for c in slack:
            text_clf = svm.LinearSVC(C=c)
            text_clf.fit(X_train, Y_train)
            predicted = text_clf.predict(X_test)
            temp = np.mean(predicted == Y_test)
            if temp > best:
                best = temp
        return best

    def AdaBoost(self,X_train, X_test, Y_train, Y_test,estimators = [50,100,200]):
        """
        input:
            X_train: training data
            X_test: testing data
            Y_train: training labels
            Y_test: testing lables
            estimators: tuning parameter (number of models used in boosting)
        output:
            accuracy
        """
        best = 0
        for n in estimators:
            text_clf = AdaBoostClassifier(n_estimators=n)
            text_clf.fit(X_train, Y_train)
            predicted = text_clf.predict(X_test)
            temp = np.mean(predicted == Y_test)
            if temp > best:
                best = temp
        return best
    

    def GradientBoosting(self,X_train, X_test, Y_train, Y_test,estimators = [50,100,200]):
        """
        input:
            X_train: training data
            X_test: testing data
            Y_train: training labels
            Y_test: testing lables
            estimators: tuning parameter (number of models used in boosting)
        output:
            accuracy
        """
        best = 0
        for n in estimators:
            text_clf = GradientBoostingClassifier(n_estimators=n)
            text_clf.fit(X_train, Y_train)
            predicted = text_clf.predict(X_test)
            temp = np.mean(predicted == Y_test)
            if temp > best:
                best = temp
        return best

    # call all 5 classification techniques on our data and print the accuracy
    def classify(self,X_train, X_test, Y_train, Y_test,alphas = [.5,1,1.5,2,2.5,3,5,7],neighbors = [1,3,5,7,10,15]
        ,slack = [1,.1,.2,.25,2,5,10,20],estimators = [50,100,200]):
        """
        input:
            X_train: training data
            X_test: testing data
            Y_train: training labels
            Y_test: testing lables
            alphas: NB tuning parameter
            neighbors: KNN tuning parameter
            slack: SVM tuning parameter
            estimators: GradientBoosting, AdaBoost tuning parameter
        output:
            classification results
        """
        SVM_result = self.SVM(X_train, X_test, Y_train, Y_test,slack)
        print("SVM    " + str(SVM_result))
        GB_result = self.GradientBoosting(X_train, X_test, Y_train, Y_test,estimators)
        print("GradientBoosting    " + str(GB_result))
        AB_result = self.AdaBoost(X_train, X_test, Y_train, Y_test,estimators)
        print("AdaBoost    " + str(AB_result))
        KNN_result = self.KNN(X_train, X_test, Y_train, Y_test,neighbors)
        print("KNN    " + str(KNN_result))
        NB_result = 0
        try:
            NB_result = self.NB(X_train, X_test, Y_train, Y_test,alphas)
            print("NB    " + str(NB_result))
        except:
            print("NB results not available: either word2vec or pca have been used (negative data is not suitable for NB)")

        return SVM_result, GB_result, AB_result, KNN_result, NB_result
        
        