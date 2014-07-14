# Digit Recognizer...

import csv
import sklearn
import numpy as np
import random
import shelve

from sklearn import svm

class KNN:
    def readFile(self,fileName):
        file = csv.reader(open(fileName, "r"))
        for row in file:
            print row
            #for field in row:
            #    print field, ", "
#algo = KNN()
#algo.readFile("train.csv")


class SVMAlgo:
    def __init__(self, trainFile, testFile):
        """
        Initializes the test and training set for the Classification to proceed
        """
        self.isPersisted = False
        self.trainX = []
        self.trainY = []
        self.testX = []

        trainData = csv.reader(open(trainFile, "r"))
        iterTrainData = iter(trainData)
        next(iterTrainData)
        for row in iterTrainData:
            #trainYRow = np.array(map(float, row[0]))
            #self.trainY.append(trainYRow)
            self.trainY.append(int(row[0]))
            trainXRow = np.array(map(int, row[1:]))
            self.trainX.append(trainXRow)
            
        #converting string list to float list
        testData =  csv.reader(open(testFile, "r"))
        iterTestData = iter(testData)
        next(iterTestData)            
        
        for row in iterTestData:
            testXRow = np.array(map(int, row))
            self.testX.append(testXRow)
        print "Initialization Complete..."
        '''
        print self.trainX
        print "Y values are: "
        print self.trainY
        print self.testX
        
        #print "Training Set: "
        #print "X"
        #print self.trainX
        #print "Y"
        #print self.trainY
        #print "Testing Set: "
        #print self.testX
        ''' 
        return
    def classifyRandomTrainingSample(self, n):
        i=0
        randomTrainingSampleX = []
        randomTrainingSampleY = []
        while i<n:
            index = random.randint(0, len(self.trainX))
            randomTrainingSampleX.append(self.trainX[index])
            randomTrainingSampleY.append(self.trainY[index])
            i = i + 1
        self.clf = svm.SVC(kernel='linear')
        print self.clf.fit(randomTrainingSampleX, randomTrainingSampleY)
        print "Classification Complete..."
        print "Support Vectors: "
        print self.clf.support_vectors_
        print "Support Vector Indices: "
        print self.clf.support_
        print "Number of Support Vectors for each of the classes: "
        print self.clf.n_support_
        return
    def classify(self):
        """
        Classification from the input training set
        """
        svmfile = 'svm_persistent_dr'
        svmkey = 'svm_poly'
        d = shelve.open(svmfile)        
        if self.isPersisted == False:
            print "Generating SVM from input..."
            self.clf = svm.SVC(kernel='poly')
            print self.clf.fit(self.trainX, self.trainY)
            d[svmkey] = self.clf
        else:
            print "Generating SVM from persistent store..."
            self.clf = d[svmkey]
        print "SVM Generated!!!"

        print "Support Vectors: "
        print self.clf.support_vectors_
        print "Support Vector Indices: "
        print self.clf.support_
        print "Number of Support Vectors for each of the classes: "
        print self.clf.n_support_
        return
    
    def test(self):
        print "Testing / Predicting the classes..."
        outputFile = open('output.csv', 'w')
        outputFileWriter = csv.writer(outputFile, delimiter=',')
        print "ImageId", ", ", "Label"
        outputFileWriter.writerow(["ImageId", "Label"])
        i = 1
        for X in self.testX:            
            # dec = self.clf.decision_function(X)
            # Number of possible classifications
            # print dec.shape[1]
            predictedOutput = self.clf.predict(X)
            predictedValue = int(predictedOutput[0])
            # Writing onto csv file
            outputRow = []
            outputRow.append(i)
            outputRow.append(predictedValue)
            outputFileWriter.writerow(outputRow)
            # Writing onto Console            
            #print i, ", ", predictedValue
            i=i+1
        return
    
# Function calls for initialization, classification and testing
algo = SVMAlgo("inputs/digitrecognizer_train.csv", "inputs/digitrecognizer_test.csv")
algo.classify()
#algo.classifyRandomTrainingSample(10000)
algo.test()

