from classifier import *
from get_dataset import *

##########################################################################
#Written By: Ahmed Abdelatty email: ahmed.abdelatty@utdallas.edu 9/6/2018
##########################################################################

def CNN_best_perform(level1_drop,level2_drop,level3_drop,level4_drop,level5_drop,batch_sizes,epochs):
    best_epoch = best_drop_perc = best_b_size = best_valid_acc = best_train_acc = best_test_acc = 0
    best_drop_perc = [.2] * 5
    for l1_drop in level1_drop:
        drop_perc[0] = l1_drop
        for l2_drop in level2_drop:
            drop_perc[1] = l2_drop
            for l3_drop in level3_drop:
                drop_perc[2] = l3_drop
                for l4_drop in level4_drop:
                    drop_perc[3] = l4_drop
                    for l5_drop in level5_drop:
                        drop_perc[4] = l5_drop
                        for b_size in batch_sizes:
                            for epoch in epochs:
                                train_acc, valid_acc,test_acc = clas.CNN(cnn_ds,batch_size=b_size,epochs=epoch
                                    ,num_classes=num_classes,dropout=dropout,drop_perc=drop_perc)
                                if valid_acc > best_valid_acc:
                                    best_valid_acc = valid_acc
                                    best_test_acc = test_acc
                                    best_train_acc = train_acc
                                    best_epoch = epoch
                                    best_drop_perc = drop_perc
                                    best_b_size = b_size
    print(" The model with the highst validation accuracy = " + str(best_valid_acc)+ "  obtained train accuracy = " + str(best_train_acc) + "  and test accuracy = "+ str(best_test_acc)
    + "\n and was trained for " + str(best_epoch) + " epoches with batch size = " + str(best_b_size) + " and drop out percentage = " + str(best_drop_perc)  )

if __name__ == '__main__':
    # classifiers tuinind parameters
    neighbors = [1,3,5,7,10,15]
    slack = [1,.1,.2,.25,2,5,10,20]
    estimators = [50,100,200]
    # define net tuinind parameters
    # drop_perc: vector of lenght 5 defining the percent of dropout at every layer
    drop_perc = [.2]*5
    level1_drop = [.25]
    level2_drop = [.25]
    level3_drop = [.35]
    level4_drop = [.4]
    level5_drop = [.45]
    batch_sizes = [64]
    epochs = [65]
    num_classes = 10
    valid_size = .2
    dropout = True
    data_loader = get_dataset()
    cnn_ds, ds = data_loader.data_preproc(use_pca=0)
    
    clas = classifier()
    # call CNN
    CNN_best_perform(level1_drop,level2_drop,level3_drop,level4_drop,level5_drop,batch_sizes,epochs)
    # call the rest of classifiers
    clas.classify(ds,neighbors = neighbors,slack = slack,estimators = estimators)
    