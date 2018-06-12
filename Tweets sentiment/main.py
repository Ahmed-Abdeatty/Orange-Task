import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from get_dataset import *
from classifier import *
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

##########################################################################
#Written By: Ahmed Abdelatty email: ahmed.abdelatty@utdallas.edu 9/6/2018
##########################################################################
best_SVM_result = 0
best_GB_result = 0
best_AB_result = 0
best_KNN_result = 0
best_NB_result = 0
best_SVM_model = ""
best_GB_model = ""
best_AB_model = ""
best_KNN_model = ""
best_NB_model = ""

# check if current model is better than the previous ones
def compare_performance(SVM_result, GB_result, AB_result, KNN_result, NB_result,model):
    global best_SVM_result,best_GB_result,best_AB_result,best_KNN_result,best_NB_result 
    global best_SVM_model,best_GB_model,best_AB_model,best_KNN_model,best_NB_model
    if SVM_result > best_SVM_result:
        best_SVM_result = SVM_result
        best_SVM_model = model
    if GB_result > best_GB_result:
        best_GB_result = GB_result
        best_GB_model = model
    if AB_result > best_AB_result:
        best_AB_result = AB_result
        best_AB_model = model
    if KNN_result > best_KNN_result:
        best_KNN_result = KNN_result
        best_KNN_model = model
    if NB_result > best_NB_result:
        best_NB_result = NB_result
        best_NB_model = model


def classify(over_sampl,tf_idf,use_idf,pca,alphas,neighbors,slack,estimators,portion):
    """
    input:
        over_sampl: string variable to indicate the name of oversampling method 
        tf_idf: boolean variable to indicate whether to use tf or not
        use_idf: boolean variable to indicate whether to use idf or not
        pca: int variable to indicate whether to use PCA or not (<=0 means no, yes otherwise)
        alphas: NB tuning parameter
        neighbors: KNN tuning parameter
        slack: SVM tuning parameter
        estimators: GradientBoosting, AdaBoost tuning parameter
        portion: which airline data to work with (None means all airlines)
    """               
    if not tf_idf:
        if pca > 0:
            return None
        else:
            message = "Preprocessing used is Word2Vec & Over Sampling method is  " + over_sampl + "  data portion  " + portion
    else:
        if use_idf:
            message = "Preprocessing used is tf-idf & Over Sampling method is  " + over_sampl + "   PCA dimension = " + str(pca) + "  data portion  " + portion
        else:
            message = "Preprocessing used is tf & Over Sampling method is  " + over_sampl + "   PCA dimension = " + str(pca) + "  data portion  " + portion
    # load dataset
    ds = get_dataset()
    X_train, X_test, Y_train, Y_test = ds.load_data(tf_idf=tf_idf,use_idf=use_idf,use_pca=pca,airway_name=portion)
    if over_sampl == "RandomOverSampler":
        X_train, Y_train = RandomOverSampler().fit_sample(X_train, Y_train)
    elif over_sampl == "SMOTE":
        X_train, Y_train = SMOTE().fit_sample(X_train, Y_train)
    elif over_sampl == "ADASYN":
        X_train, Y_train = ADASYN().fit_sample(X_train, Y_train)
    clas = classifier()
    print(message)
    SVM_result, GB_result, AB_result, KNN_result, NB_result = clas.classify(X_train, X_test, Y_train, Y_test)
    compare_performance(SVM_result, GB_result, AB_result, KNN_result, NB_result,message)




if __name__ == "__main__":

    # define tuning parameters
    # perform classification on the entire data if None or the portion regards the stated airline
    data_portion = ["All","@JetB","@Americ","@united","@VirginA","@Southw","@USAir"]
    over_sampl_opt = ["None","RandomOverSampler","SMOTE","ADASYN"]
    #pca dimensions
    pca_dim = [0]
    # perform either tf-idf, tf, or word2vec 
    tf_idf_opt=[True,False]
    idf_opt=[True,False]
    # classifiers tuinind parameters
    alphas = [.5,1,1.5,2,2.5,3,5,7]
    neighbors = [1,3,5,7,10,15]
    slack = [1,.1,.2,.25,2,5,10,20]
    estimators = [50,100,200]
    for portion in data_portion:
        for over_sampl in over_sampl_opt:
            for pca in pca_dim:
                for tf_idf in tf_idf_opt:
                    if tf_idf == True:
                        for use_idf in idf_opt:
                            classify(over_sampl,tf_idf,use_idf,pca,alphas,neighbors,slack,estimators,portion)
                    else:
                        classify(over_sampl,tf_idf,False,pca,alphas,neighbors,slack,estimators,portion)
                        
    print("Highest SVM accuracy  " + str(best_SVM_result) + " best model  " + best_SVM_model)
    print("Highest GB accuracy  " + str(best_GB_result) + " best model  " + best_GB_model)
    print("Highest AB accuracy  " + str(best_AB_result) + " best model  " + best_AB_model)
    print("Highest KNN accuracy  " + str(best_KNN_result) + " best model  " + best_KNN_model)
    print("Highest NB accuracy  " + str(best_NB_result) + " best model  " + best_NB_model)



"""
raw_data = {'classifier_name': ['SVM', 'GB', 'AB', 'KNN', 'NB'],
        'tf_idf': [80.26, 76.81, 76.63, 74.76, 77.56],
        'tf': [80.6, 76.74, 76.16, 72.06, 79.06],
        'word2vec': [74.72, 74.52, 71.45, 71.99, 0]}
df = pd.DataFrame(raw_data, columns = ['classifier_name', 'tf_idf', 'tf', 'word2vec'])
# Setting the positions and width for the bars
pos = list(range(len(df['tf_idf']))) 
width = 0.25 
    
# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with tf-idf data,
# in position pos,
plt.bar(pos, 
        #using df['tf-idf'] data,
        df['tf_idf'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in classifier_name
        label=df['classifier_name'][0]) 

# Create a bar with tf data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using df['tf'] data,
        df['tf'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label the second value in classifier_name
        label=df['classifier_name'][1]) 

# Create a bar with word2vec data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], 
        #using df['word2vec'] data,
        df['word2vec'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#FFC222', 
        # with label the third value in classifier_name
        label=df['classifier_name'][2]) 

# Set the y axis label
ax.set_ylabel('Accuracy')

# Set the chart's title
ax.set_title('Comparison between Tf-idf, Tf, and Word2Vec accuracy')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['classifier_name'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, 100] )

# Adding the legend and showing the plot
plt.legend(['Tf-idf', 'Tf', 'Word2Vec'], loc='upper right')
plt.grid()
plt.show()
"""