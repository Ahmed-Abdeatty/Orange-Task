from classifier import *
from get_dataset import *

##########################################################################
#Written By: Ahmed Abdelatty email: ahmed.abdelatty@utdallas.edu 9/6/2018
##########################################################################
if __name__ == '__main__':
    # classifiers tuinind parameters
    neighbors = [1,3,5,7,10,15]
    slack = [1,.1,.2,.25,2,5,10,20]
    estimators = [50,100,200]
    # define net tuinind parameters
    # drop_perc: vector of lenght 5 defining the percent of dropout at every layer
    drop_perc = [.2]*5
    level1_drop = [.2,.25,.3]
    level2_drop = [.2,.25,.3]
    level3_drop = [.25,.3,35]
    level4_drop = [.3,.35,.4]
    level5_drop = [.35,.4,.5]
    batch_sizes = [32,64,128]
    epochs = [32,65,100,150]
    num_classes = 10
    valid_size = .2
    dropout = True
    data_loader = get_dataset()
    cnn_ds, ds = data_loader.data_preproc(use_pca=0)
    
    clas = classifier()
    # call CNN
    for l1_drop in level1_drop:
        for l1_drop in level1_drop:
            for l1_drop in level1_drop:
                for l1_drop in level1_drop:
                    for l1_drop in level1_drop:
        pass
    # call the rest of classifiers
    clas.classify(ds,neighbors = neighbors,slack = slack,estimators = estimators)
    