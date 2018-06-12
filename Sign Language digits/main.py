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
        for l2_drop in level2_drop:
            for l3_drop in level3_drop:
                for l4_drop in level4_drop:
                    for l5_drop in level5_drop:
                        for b_size in batch_sizes:
                            for epoch in epochs:
                                drop_perc[0] = l1_drop
                                drop_perc[1] = l2_drop
                                drop_perc[2] = l3_drop
                                drop_perc[3] = l4_drop
                                drop_perc[4] = l5_drop
                                train_eval, valid_eval,test_eval = clas.CNN(cnn_ds,batch_size=b_size,epochs=epoch
                                    ,num_classes=num_classes,dropout=dropout,drop_perc=drop_perc)
                                print(str(train_eval) + "  " + str(valid_eval) + "  " + str(test_eval) )
    # call the rest of classifiers
    clas.classify(ds,neighbors = neighbors,slack = slack,estimators = estimators)
    