# Summary
In this repository, I applied multiple classification techniques including SVM, Convolution Neural Networks, Naïve Bayes, K Nearest Neighbors, GradientBoosting, and AdaBoost. Moreover, many text feature extractors have been used: Tf, Tf-idf, and Word2Vec. I used both the classification techniques and the feature extractors to classify two datasets: 1st is the set of tweets about US airline companies and their sentiment, 2nd is the Turkish sign language digits. The sentiment analysis dataset was imbalanced, so I applied over sampling methods (SMOTE, ADASYN) to adjust the class distribution. Furthermore, both datasets were high dimensional, so I useed PCA for dimensionality reduction. For the sentiment analysis task, the best test accuracy obtained was 80.6% and was obtained by SVM. On the other hand, the best classification test accuracy I got for the Turkish sign language dataset was 98.06% and was obtained by Convolution Neural Networks.

# Datasets description
## US airline tweets
 This dataset contains tweets about six US airline companies, each of which is labelled as: positive, negative, and neutral. The total number of tweets is 14,640.

## Turkish sign language digits

 This dataset contains images of the Turkish sign language digits. The dataset has 10 class for digits from 0 to 9. The total number of images is 2,062 and labels are almost evenly distributed.

# Libraries used
1. sklearn: to use classifers like SVM, NB, KNN, etc on both datasets. 
2. textblob: I used the pretrained TextBlob sentiment classifier.
3. keras & tensorflow: to train CNN over the sign language digits datasets.

# Runing Requirements
To run this classification tasks you need
1. Python version 3.52 or latter.
2. you need to have the following Python packages installed (sklearn, textblob, numpy, keras, and bs4)

# How to run
Both classification tasks have a main.py file that can be run directly. However, you may need to edit the list of tuning parameters.

In order to run the Word2Vec portion of the code you need to download Glove 100 dim (glove.6B.100d.txt) (can be downloaded from https://worksheets.codalab.org/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/) and save it in the same directory as the .py files
