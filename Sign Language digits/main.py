from classifier import *
from get_dataset import *
import itertools
##########################################################################
#Written By: Ahmed Abdelatty email: ahmed.abdelatty@utdallas.edu 9/6/2018
##########################################################################

def CNN_best_perform(drop_percs,batch_sizes,epochs):
    best_epoch = best_drop_perc = best_b_size = best_valid_acc = best_train_acc = best_test_acc = 0
    best_drop_perc = [.2] * 5
    for drop_perc in drop_percs:
        for b_size in batch_sizes:
            for epoch in epochs:
                train_acc, valid_acc,test_acc = clas.CNN(cnn_ds,batch_size=b_size,epochs=epoch
                    ,num_classes=num_classes,dropout=dropout,drop_perc=drop_perc)
                #print(str(train_acc) + "   " + str(valid_acc) + "   " + str(test_acc))
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
    level1_drop = [.2,.25,.28,.3]
    level2_drop = [.2,.25,.28,.3]
    level3_drop = [.5,.3,.35,.4]
    level4_drop = [.3,.35,.4,.45]
    level5_drop = [.35,.4,.45,5]
    # get all combinations 
    drop_percs = [level1_drop,level2_drop,level3_drop,level4_drop,level5_drop]
    drop_percs = list(itertools.product(*drop_percs))
    batch_sizes = [16,32,64,128]
    epochs = [32,65,100,150]
    num_classes = 10
    valid_size = .2
    dropout = True
    data_loader = get_dataset()
    cnn_ds, ds = data_loader.data_preproc(use_pca=0)
    
    clas = classifier()
    # call CNN
    CNN_best_perform(drop_percs,batch_sizes,epochs)
    # call the rest of classifiers
    clas.classify(ds,neighbors = neighbors,slack = slack,estimators = estimators)
    