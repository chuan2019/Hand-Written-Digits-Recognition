#-------------------------------------------------------------------------------
# Name:        digit_recognizer
# Purpose:     This is .py file includes all my python code for solving the
#              hand-written digits recognizing problem on Kaggle Competitions.
#              As the first attempts, I tried both Logistic Regression, and SVM
#              methods. Performance of the two algorithms was evaluated with
#              ROC curves (One-VS-Rest) and AUC (area under the curve)
#
# Author:      Chuan
#
# Created:     25/11/2014
# Copyright:   (c) Chuan 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import csv
import numpy as np
import math

class digit_recognizer(object):
    """
    digit_recognizer: This is my solution to the Kaggle data-science competition
      Digit Recognizer: https://www.kaggle.com/c/digit-recognizer
      As my first attempt, I use both Logistic Regression and SVM methods. To
      evaluate the performance of the two models, both ROC curve and AUC are
      used.
    """
##______________________________________________________________________________
    def __init__(self, file_name):
        """
        method    :  __init__(self, file_name)
        Purpose   :  This is the initializer of the class digit_recognizer. In
                     this initializer, the raw data will be loaded, and splitted
                     into a training data set and a testing data set for
                     training and evaluating the performance of the model.
        Input     :  file_name, str: a string containing the path and name of
                     the raw data file. It has to be a .csv file with the first
                     column as the targets/keys of the corresponding features
                     represented by the rest part of each row of the data file.
        Output    :  train_x: list of arrays
                     train_y: list of integers (0~9)
                     test_x: list of arrays
                     test_y: list of integers (0~9)
                     X: list of arrays (for prediction)
        """
        try:
            self.ftrain_name = file_name[0]
            self.ftest_name = file_name[1]
        except ValueError:
            print "Please input the paths and file names for both training data \
            set and testing data set!!"

        self.file_name = file_name
        self.data, self.images, self.target = self.__load_data__(self.ftrain_name, \
        skip_head=True, train=True)
        idx = self.create_data_partition(self.data)

        self.train_x = self.data[idx[0]]
        self.train_y = self.target[idx[0]]
        self.test_x = self.data[idx[1]]
        self.test_y = self.target[idx[1]]

        self.X, self.ximages, _ = self.__load_data__(self.ftest_name, \
        skip_head=True, train=False)
##____________________________________________________________________________##
    def __load_data__(self, file_name, skip_head=True, train=True):
        """
        Purpose: This method opens a .csv (comma separated values) file, and
                 loads the hand-written digit data into the memory.
        Input  : file_name: str, the path and name of the .csv file
                 skip_head: boolean, if True, then skip the first line of the
                            file, which is the file header; if False, it implies
                            that the file does not have a header
                 train    : boolean, if True, then the first element in each row
                            is the target (i.e., the digit, the rest image
                            information represents); if False, it means that the
                            .csv file is for purely testing/predicting, so there
                            is no key (target).
        Output : digits   : numpy array, of the shape [n_samples, nrow*ncol].
                            Usually, nrow = ncol. It contains pixel information
                            of each image as features.
                 images   : numpy array, of the shape [n_samples, [nrow,
                            ncol]]
                 target   : array of the shape [n_samples], if train is True,
                            it contains all the keys corresponding to each
                            image; if train is False, it is an empty list.
        """
        print "Loading ", file_name,
        with open(file_name, 'rb') as in_file:
            rawdata = csv.reader(in_file, delimiter=',', quotechar='|')
            digits = []
            images = []
            target = []
            count = 0
            for rows in rawdata:
                if count % 10000 == 0:
                    print ".",
                if skip_head:
                    skip_head = False
                    continue
                if train:
                    target.append(float(rows[0]))
                    digit = np.array(map(float, rows[1:]))
                    digits.append(digit)
                    nrow = math.sqrt(digit.shape[0])
                    ncol = nrow
                    images.append(digit.reshape((nrow, ncol)))
                else:
                    digit = np.array(map(float, rows))
                    digits.append(digit)
                    nrow = math.sqrt(digit.shape[0])
                    ncol = nrow
                    images.append(digit.reshape((nrow, ncol)))
                count += 1
            print "\n", file_name, "is loaded.\n"
            return (np.array(digits), np.array(images), np.array(target))
##____________________________________________________________________________##
    def create_data_partition(self, data, in_train=0.6, seed=5425):
        """
        Method    :  create_data_partition(data, times=1, in_train=0.6)
        Purpose   :  Create partition(s) of the given dataset for training and
                     testing
        Input     :  data  : non-empty dataset (list, array, numpy array, or
                             pandas series or data.frame etc.)
                     in_train: a rational number in [0, 1] indicating the
                             portion of the data for training
                     seed  : a natural number (default value=5425) defining the
                             seed for generating sequence of random numbers
        Output    :  a tuple of the two indices for the observations partitioned
                     into the training set and the testing set respectively
        ________________________________________________________________________
        Comments  :  I am mimicing R to create a function to generate a
                     partition of the data set into training set and test set.
                     An alternative is to use the following function:

                        sklearn.cross_validation.train_test_split(*arrays, \
                        **options)

                     check the website: scikit-learn.org for detailed
                     instructions.
        """
        import random as rd
        print "Start splitting the training set into two parts ..."
        sample_size = data.shape[0]
        rd.seed(seed)
        train = rd.sample(xrange(sample_size), int(in_train*sample_size))
        test = filter(lambda x: x not in set(train), range(sample_size))
        print "Dataset partitioning is done!"
        return (train, test)
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    def vectorizer2(func):
        """
        Function  :  vectorizer2(func)
        Purpose   :  This function is used to wrap the method 'logfit' which
                     take numpy 1d or 2d array X and 1d array y as input. As
                     some other methods output list of arrays, and then we need
                     to feed them to the method mentioned above as the input, we
                     might not convert them into ndarrays. So this decorator
                     is used to check if the input positional variables are
                     numpy arrays or not, if not then the checker function will
                     convert them into numpy arrays.
        Input     :  func, the method taking two numpy arrays (and other
                     variables/parameters) as input
        Output    :  checker
        """
        def checker(self, X, y, C=1e5, tol=1e-1):
            import numpy as np
            X = np.array(X)
            y = np.array(y)
            ret = func(self, X, y, C=1e5, tol=1e-1)
            return ret
        return checker
##____________________________________________________________________________##
    @vectorizer2
    def logfit(self, X, y, C=1e5, tol=1e-1):
        """
        Method    :  logfit(X, y, C=1e5, tol=1e-1)
        Input     :  X: array of the shape [n_samples, nrow*ncol], it contains
                        features of every sample.
                     y: array of the shape [n_samples], it contains targets
                        (keys) of every sample.
                     C: inverse of regularization strength, must be a positive
                        float, and default value is 1.0.
                     tol: tolerance for stopping criteria, must be a positive
                        float too.
        Output    :  self, the estimator object
        """
        from sklearn import metrics
        from sklearn import preprocessing
        from sklearn import multiclass as mc
        from sklearn import linear_model as lm

        print "Start training the Logistic Regression model ..."
        # Binarize the output
        y = preprocessing.label_binarize(y, classes=range(10))
        classifier = mc.OneVsRestClassifier(lm.LogisticRegression(C=C, tol=tol))
        model = classifier.fit(X, y)
        print "Model training is done!"
        return model
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    def vectorizer1(func):
        """
        Function  :  vectorizer1(self, func)
        Purpose   :  This function is used to wrap the method 'make_pred' which
                     takes numpy 1d or 2d array X as input. As some other
                     methods output list of arrays, and then we need to feed
                     them to the method mentioned above as input we might not
                     convert them into ndarrays. So this decorator is used to
                     check if the input positional variables are numpy arrays or
                     not, if not then the checker function will convert them
                     into numpy arrays.
        Input     :  func, the method taking one numpy arrays (and other
                     variables/parameters) as input
        Output    :  checker
        """
        def checker(self, X, model):
            import numpy as np
            X = np.array(X)
            ret = func(self, X, model)
            return ret
        return checker
##____________________________________________________________________________##
    @vectorizer1
    def make_pred(self, X, model):
        """
        Method    :  make_pred(X, model)
        Input     :  X: array of the shape [n_samples, nrow*ncol], it contains
                        all the features of every sample from the testing set
                     model: estimator object, it must be fitted/trained,
                        otherwise, an error message would be raised
        Output    :  y: array of the shape [n_samples], it contains the digits
                        predicted by the fitted model
        """
        import matplotlib.pyplot as plt
        import pylab as pl

        print "Start recognizing the hand-written digits ..."
        pred_y = model.predict(X)
        y = self.__debinarize__(pred_y)
        if -1 in y:
            score_y = model.decision_function(X)
            neg1_idx = [i for i,j in enumerate(y) if j == -1]
            plt.figure(figsize=(15,3), dpi=1000)
            count = 0
            for idx in neg1_idx:
                y[idx] = list(score_y[idx]).index(max(score_y[idx]))
                if count < 10:
                    plt.subplot(1,10,(count+1))
                    pl.imshow(X[idx].reshape((28,28)), cmap=pl.cm.gray_r)
                    plt.title('Predict:{0}'.format(y[idx]))
                count += 1
            plt.savefig('../figures/Diverge.png')
        print "All hand-written digits are recognized!"
        return y
##____________________________________________________________________________##
    def __debinarize__(self, binarized_array):
        """
        Method    :  __debinarize__(self, binarized_array)
        Input     :  binarized_array: array of the shape [n_samples, n_classes]
                                      it contains the targets/predictions of an
                                      multi-class prediction problem. Each row
                                      only has one element being 1, and all
                                      others are 0's.
        Output    :  target : array of the shape [n_samples], it contains the
                              class labels of each sample. The elements in
                              target must be non-negative integers.
        """
        target = []
        for rows in binarized_array:
            if sum(rows) == 0:
                target.append(-1)
            else:
                target.append(list(rows).index(1))
        target = np.array(target)
        return target
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    def roc_and_auc(self, pred_prob, pred_key):
        """
        Method    :  roc_and_auc(self, pred_prob, pred_key)
        Input     :  pred_prob: array of shape [n_samples, n_classes], contains
                                the probability scores for each sample. E.g. the
                                probability scores for the k-th sample is
                                [0.1, 0.4, 0.8, 0.1, -0.1, -0.2, -0.5, 0.1, 0.3,
                                0.2], then we say the k-th sample is most likely
                                to be the digit 2.
                     pred_key:  array of shape [n_samples], contains the true
                                digit that the k-th sample respresents.
        Output    :  roc_auc: array of shape [n_samples, n_classes], contains
                                the auc values for each classifier (One-VS-Rest)
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn import preprocessing
        import matplotlib.pyplot as plt

        print "Plotting the ROC curves ..."
        if len(np.array(pred_key).shape) == 1:
            pred_key = preprocessing.label_binarize(np.array(pred_key), \
            classes=range(10))
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = pred_key.shape[1]
        for cidx in range(n_classes):
            fpr[cidx], tpr[cidx], _ = roc_curve(pred_key[:, cidx], \
            pred_prob[:, cidx])
            roc_auc[cidx] = auc(fpr[cidx], tpr[cidx])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(pred_key.ravel(), \
        pred_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        plt.figure(figsize=(10,10), dpi=1500)
        plt.plot(fpr["micro"], tpr["micro"], \
        label='micro-average ROC curve (area = {0:0.2f})' \
        ''.format(roc_auc["micro"]))
        for idx in range(n_classes):
            plt.plot(fpr[idx], tpr[idx], label='ROC curve of class {0} \
            (area = {1:0.2f})' ''.format(idx, roc_auc[idx]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
##        plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
        plt.title('Multi-Class ROC Curves for Logistic Regression')
        plt.savefig('../figures/LogROC.png')
        print "ROC curves are plotted."
        return roc_auc
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
    def output_pred(self, y, ofile):
        """
        Method    :  output_pred(self, y, ofile)
        Input     :  y: array of the shape [n_samples], contains the recoganized
                        digits
                     ofile: str, contains both the path and name of the .csv
                        file that will be used to output the recoganized digits
        Output    :  boolean, if True, output is successful; otherwise output
                        failed.
        """
        n_samples = len(y)
        print "Outputing recoganized digits ",
        with open(ofile, 'wb') as fout:
            output = csv.writer(fout, delimiter=',')
            count = 0
            for rows in y:
                if count % 10000 == 0:
                    print ".",
                output.writerow([rows])
            print "\nOutputing is done!"
        return True
##============================================================================##
def main():
    """
    The class digit_recognizer is designed to be an extendable module for
    hand-written digits recognition.
    """
    dig_rec = digit_recognizer(['../data/train.csv','../data/test.csv'])
    modfit1 = dig_rec.logfit(dig_rec.train_x, dig_rec.train_y)
    pred_y  = dig_rec.make_pred(dig_rec.X, modfit1)
    dig_rec.output_pred(pred_y, '../data/pred.csv')
    score_y = modfit1.decision_function(dig_rec.test_x)
    auc = dig_rec.roc_and_auc(score_y, dig_rec.test_y)

if __name__ == '__main__':
    main()
