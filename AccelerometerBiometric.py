# SVM (Accelerometer Biometric)...

import csv
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import random
import MySQLdb
import _mysql
import shelve

import scipy
import scipy.fftpack
import pylab

from scipy import pi
from decimal import *
from sklearn import svm

class Config:
    
    def getDBConnection(self):
        # Socket location of Mysql-TukoDB: /tmp/mysql.sock
        # Port Number: 3306
        unix_socket = '/tmp/mysql.sock'
        #db = _mysql.connect(host="localhost",port=3306,user="root",passwd="ramsri",db="AccBiometric",read_default_file="/etc/my.cnf")
        db = MySQLdb.connect(host="localhost", user="root",port=3306, passwd="ramsri", db="AccBiometric",  read_default_file="/etc/my.cnf")
        return db
    
    def loadTrainData(self, trainFile):
        # Open database connection
        db = self.getDBConnection()
        trainData = csv.reader(open(trainFile, "r"))
        iterTrainData = iter(trainData)
        next(iterTrainData)
        # adjust the columns for the row
        i=0
        for row in iterTrainData:
            # prepare a cursor object using cursor() method
            cursor = db.cursor()
            #SQL query to INSERT a record into the table trainData.
            cursor.execute('''INSERT into trainData (T, X, Y, Z, D) values (%s, %s, %s, %s, %s)''', (row[0], row[1], row[2], row[3], row[4]))
            # Commit your changes in the database
            i += 1
            if i%1000000==0:
                print i, "Records inserted Succesfully!!!"
        db.commit()
        # disconnect from server
        db.close()
        print "Loading Training data Complete!!!"
        return
    
    def loadTestData(self, testFile):
        # Open database connection
        db = self.getDBConnection()
        testData = csv.reader(open(testFile, "r"))
        iterTestData = iter(testData)
        next(iterTestData)
        # adjust the columns for the row
        i=0
        for row in iterTestData:
            # prepare a cursor object using cursor() method
            cursor = db.cursor()
            #SQL query to INSERT a record into the table testData.
            cursor.execute('''INSERT into testData (T, X, Y, Z, S) values (%s, %s, %s, %s, %s)''', (row[0], row[1], row[2], row[3], row[4]))
            # Commit your changes in the database
            i += 1
            if i%100000==0:
                print i, "Records inserted Succesfully!!!"
        db.commit()
        # disconnect from server
        db.close()
        print "Loading Test data Complete!!!"
        return
    
    def comparePrecision(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute("SELECT d FROM DoublePrecision")
        item = cursor.fetchone()
        print item
        print item[0]
        print Decimal(item[0])
        cursor.execute("SELECT d FROM DecimalPrecision")
        item = cursor.fetchone()
        print item
        print item[0]
        print Decimal(item[0])
        return

    def loadFeatureData(self):
        iterations = 0
        # All the CRUD operations over database
        db = self.getDBConnection()
        # prepare a cursor object using cursor() method
        cursor = db.cursor()
        # Select qSQL with id=4.
        cursor.execute("SELECT DISTINCT(D) FROM trainData")
        # getting only one record (enough for getting count)
        devices = cursor.fetchall()
        for device in devices:
            # Counting the number of devices
            deviceNumber = device[0]
            cursor.execute("SELECT count(*) FROM trainData WHERE D=%s", (deviceNumber))
            deviceCount = cursor.fetchone()[0]
            sampleSize = 100
            noTrainingSamples = 100;
            while noTrainingSamples>0:
                limitOffset = random.randint(0, deviceCount-sampleSize)
                cursor.execute('''
                                SELECT avg(X), avg(Y), avg(Z), avg(X*Y)-avg(X)*avg(Y), avg(Y*Z)-avg(Y)*avg(Z), avg(X*Z)-avg(X)*avg(Z), variance(X), variance(Y), variance(Z), D 
                                FROM (SELECT X, Y, Z, D FROM trainData where D=%s limit %s, %s) as sampleTable''', (deviceNumber, limitOffset, sampleSize))
                resultsSet = cursor.fetchone()

                cursor.execute("SELECT max(X) as MedX FROM (SELECT X FROM (select X from trainData where D=%s limit %s, %s) as xTable ORDER BY X limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medX = cursor.fetchone()
                cursor.execute("SELECT max(Y) as MedX FROM (SELECT Y FROM (select Y from trainData where D=%s limit %s, %s) as yTable ORDER BY Y limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medY = cursor.fetchone()
                cursor.execute("SELECT max(Z) as MedZ FROM (SELECT Z FROM (select Z from trainData where D=%s limit %s, %s) as zTable ORDER BY Z limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medZ = cursor.fetchone()
                
                
                # Getting a precision of about 20 (decimal place)
                getcontext().prec = 20;

                MeanX = Decimal(resultsSet[0])/Decimal(1)
                MeanY = Decimal(resultsSet[1])/Decimal(1)
                MeanZ = Decimal(resultsSet[2])/Decimal(1)
                CoVarXY = Decimal(resultsSet[3])/Decimal(1)
                CoVarYZ = Decimal(resultsSet[4])/Decimal(1)
                CoVarXZ = Decimal(resultsSet[5])/Decimal(1)
                VarX = Decimal(resultsSet[6])/Decimal(1)
                VarY = Decimal(resultsSet[7])/Decimal(1)
                VarZ = Decimal(resultsSet[8])/Decimal(1)
                MedX = medX[0]
                MedY = medY[0]
                MedZ = medZ[0]
                
                # test values from data base
                '''
                print 'Device: ', deviceNumber
                print 'Limit Offset: ', limitOffset
                print 'AVG X: ', MeanX
                print 'AVG Y: ', MeanY
                print 'AVG Z: ', MeanZ
                print 'COV XY: ', CoVarXY
                print 'COV YZ: ', CoVarYZ
                print 'COV XZ: ', CoVarXZ
                print 'VAR X: ', VarX
                print 'VAR Y: ', VarY
                print 'VAR Z: ', VarZ
                print 'MED X: ', MedX
                print 'MED Y: ', MedY
                print 'MED Z: ', MedZ
                print ''
                '''
                # storing results into featureData
                cursor.execute('''INSERT INTO featureData (VarX, VarY,	VarZ, MeanX,
                                MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, D)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                               (VarX, VarY, VarZ, MeanX, MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, deviceNumber))
                noTrainingSamples -= 1
                # end of sampleTrainingData
            print "Device: ", deviceNumber, ", Sampling training data complete"
            #end of device loop
        # commiting the insertion on table featureData
        db.commit()
        print 'Feature data extraction complete!!!'
        return

    def loadCrossValidationFeatureData(self):
        iterations = 0
        # All the CRUD operations over database
        db = self.getDBConnection()
        # prepare a cursor object using cursor() method
        cursor = db.cursor()
        # Select qSQL with id=4.
        cursor.execute("SELECT DISTINCT(D) FROM trainData")
        # getting only one record (enough for getting count)
        devices = cursor.fetchall()
        for device in devices:
            # Counting the number of devices
            deviceNumber = device[0]
            cursor.execute("SELECT count(*) FROM trainData WHERE D=%s", (deviceNumber))
            deviceCount = cursor.fetchone()[0]
            sampleSize = 100
            noTrainingSamples = 10;
            while noTrainingSamples>0:
                limitOffset = random.randint(0, deviceCount-sampleSize)
                cursor.execute('''
                                SELECT avg(X), avg(Y), avg(Z), avg(X*Y)-avg(X)*avg(Y), avg(Y*Z)-avg(Y)*avg(Z), avg(X*Z)-avg(X)*avg(Z), variance(X), variance(Y), variance(Z), D 
                                FROM (SELECT X, Y, Z, D FROM trainData where D=%s limit %s, %s) as sampleTable''', (deviceNumber, limitOffset, sampleSize))
                resultsSet = cursor.fetchone()

                cursor.execute("SELECT max(X) as MedX FROM (SELECT X FROM (select X from trainData where D=%s limit %s, %s) as xTable ORDER BY X limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medX = cursor.fetchone()
                cursor.execute("SELECT max(Y) as MedX FROM (SELECT Y FROM (select Y from trainData where D=%s limit %s, %s) as yTable ORDER BY Y limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medY = cursor.fetchone()
                cursor.execute("SELECT max(Z) as MedZ FROM (SELECT Z FROM (select Z from trainData where D=%s limit %s, %s) as zTable ORDER BY Z limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medZ = cursor.fetchone()
                
                
                # Getting a precision of about 20 (decimal place)
                getcontext().prec = 20;

                MeanX = Decimal(resultsSet[0])/Decimal(1)
                MeanY = Decimal(resultsSet[1])/Decimal(1)
                MeanZ = Decimal(resultsSet[2])/Decimal(1)
                CoVarXY = Decimal(resultsSet[3])/Decimal(1)
                CoVarYZ = Decimal(resultsSet[4])/Decimal(1)
                CoVarXZ = Decimal(resultsSet[5])/Decimal(1)
                VarX = Decimal(resultsSet[6])/Decimal(1)
                VarY = Decimal(resultsSet[7])/Decimal(1)
                VarZ = Decimal(resultsSet[8])/Decimal(1)
                MedX = medX[0]
                MedY = medY[0]
                MedZ = medZ[0]
                
                # storing results into crossValidationFeatureData
                cursor.execute('''INSERT INTO crossValidationFeatureData (VarX, VarY,	VarZ, MeanX,
                                MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, D)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                               (VarX, VarY, VarZ, MeanX, MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, deviceNumber))
                noTrainingSamples -= 1
                # end of sampleTrainingData
            print "Device: ", deviceNumber, ", Sampling training data complete"
            #end of device loop
        # commiting the insertion on table featureData
        db.commit()
        print 'Cross Validation Feature data extraction complete!!!'
        return

    def loadTestFeatureData(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute("SELECT DISTINCT(S) FROM testData")
        # getting only one record (enough for getting count)
        devices = cursor.fetchall()
        for device in devices:
            # Counting the number of devices
            deviceNumber = device[0]
            cursor.execute("SELECT count(*) FROM testData WHERE S=%s", (deviceNumber))
            deviceCount = cursor.fetchone()[0]
            sampleSize = 100
            noTrainingSamples = 1;
            while noTrainingSamples>0:
                limitOffset = random.randint(0, deviceCount-sampleSize)
                cursor.execute('''
                                SELECT avg(X), avg(Y), avg(Z), avg(X*Y)-avg(X)*avg(Y), avg(Y*Z)-avg(Y)*avg(Z), avg(X*Z)-avg(X)*avg(Z), variance(X), variance(Y), variance(Z), S 
                                FROM (SELECT X, Y, Z, S FROM testData where S=%s limit %s, %s) as sampleTable''', (deviceNumber, limitOffset, sampleSize))
                resultsSet = cursor.fetchone()

                cursor.execute("SELECT max(X) as MedX FROM (SELECT X FROM (select X from testData where S=%s limit %s, %s) as xTable ORDER BY X limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medX = cursor.fetchone()
                cursor.execute("SELECT max(Y) as MedX FROM (SELECT Y FROM (select Y from testData where S=%s limit %s, %s) as yTable ORDER BY Y limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medY = cursor.fetchone()
                cursor.execute("SELECT max(Z) as MedZ FROM (SELECT Z FROM (select Z from testData where S=%s limit %s, %s) as zTable ORDER BY Z limit %s) sampleTable", (deviceNumber, limitOffset, sampleSize, sampleSize/2))
                medZ = cursor.fetchone()
                
                
                # Getting a precision of about 20 (decimal place)
                getcontext().prec = 20;

                MeanX = Decimal(resultsSet[0])/Decimal(1)
                MeanY = Decimal(resultsSet[1])/Decimal(1)
                MeanZ = Decimal(resultsSet[2])/Decimal(1)
                CoVarXY = Decimal(resultsSet[3])/Decimal(1)
                CoVarYZ = Decimal(resultsSet[4])/Decimal(1)
                CoVarXZ = Decimal(resultsSet[5])/Decimal(1)
                VarX = Decimal(resultsSet[6])/Decimal(1)
                VarY = Decimal(resultsSet[7])/Decimal(1)
                VarZ = Decimal(resultsSet[8])/Decimal(1)
                MedX = medX[0]
                MedY = medY[0]
                MedZ = medZ[0]
                
                # storing results into crossValidationFeatureData
                cursor.execute('''INSERT INTO testFeatureData (VarX, VarY, VarZ, MeanX,
                                MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, S)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                               (VarX, VarY, VarZ, MeanX, MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, deviceNumber))
                noTrainingSamples -= 1
                # end of sampleTrainingData
            print "Sequence: ", deviceNumber, ", Sampling testing data complete"
            #end of device loop
        # commiting the insertion on table featureData
        db.commit()
        print 'Test Feature data extraction complete!!!'
        return

    def getOriginalTrainData(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute('''SELECT SQRT(X*X + Y*Y + Z*Z)
                            FROM trainData limit 1000''')
        resultsSet = cursor.fetchall()
        return resultsSet

    def getDistinctDevices(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute('''SELECT distinct(D)
                            FROM trainData''')
        resultsSet = cursor.fetchall()
        return resultsSet

    def getDistinctSequences(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute('''SELECT distinct(S)
                            FROM testData''')
        resultsSet = cursor.fetchall()
        return resultsSet
    
    def getOriginalTrainDataForDevice(self, deviceNumber):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute('''SELECT SQRT(X*X + Y*Y + Z*Z)
                            FROM trainData where D=%s''', (deviceNumber))
        resultsSet = cursor.fetchall()
        return resultsSet

    def getTrainFeatureData(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute('''SELECT VarX, VarY, VarZ, MeanX,
                            MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, D
                            FROM featureData''')
        resultsSet = cursor.fetchall()
        return resultsSet

    def getCrossValidationFeatureData(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute('''SELECT VarX, VarY, VarZ, MeanX,
                            MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, D
                            FROM crossValidationFeatureData''')
        resultsSet = cursor.fetchall()
        return resultsSet

    def getTestFeatureData(self):
        db = self.getDBConnection()
        cursor = db.cursor()
        cursor.execute('''SELECT VarX, VarY, VarZ, MeanX,
                            MeanY, MeanZ, MedX, MedY, MedZ, CoVarXY, CoVarYZ, CoVarXZ, S
                            FROM testFeatureData''')
        resultsSet = cursor.fetchall()
        return resultsSet

    # Gets the questions data - to process along with the train data
    def getQuestionsData(self):
        listQuestions = []
        testData = csv.reader(open("inputs/questions.csv", "r"))
        iterTestData = iter(testData)
        next(iterTestData)
        for row in iterTestData:
            listQuestions.append(row[:])
        return listQuestions

    
class AccBiometric:
    # Cross Validating the feature data with the cross Vadiation feature data
    # initialize the load, train, crossvalidate parameters to false
    # defining that the data is not yet loaded
    def __init__(self):
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testS = []
        self.crossX = []
        self.crossY = []
        self.questionY = []
        self.isTestLoaded = False
        self.isQuestionLoaded = False
        self.isCrossValidationLoaded = False
        self.isTrainLoaded = False
        self.isSVMGenerated = False
        self.isSVMPersisted = False
        self.isRFGenerated = False
        self.isRFPersisted = False
        #self.clf = svm.SVC(kernel='poly')
        self.clf = svm.LinearSVC(multi_class='ovr')
        self.rf = RandomForestClassifier(n_estimators=20, criterion="entropy")
        return

    # Load test data if not loaded already
    def loadTestData(self):
        if self.isTestLoaded==False:
            print "Test data is Loading..."
            resultsSet = Config().getTestFeatureData()
            for row in resultsSet:
                self.testX.append(row[:-1])
                self.testS.append(int(row[-1:][0]))
            self.isTestLoaded = True
        print "Test Data Loaded!!!"        
        return
    
    # Load train data if not loaded already
    def loadTrainData(self):
        if self.isTrainLoaded==False:
            print "Train data is Loading..."            
            resultsSet = Config().getTrainFeatureData()
            for row in resultsSet:
                self.trainX.append(row[:-1])
                self.trainY.append(int(row[-1:][0]))
            self.isTrainLoaded=True
        print "Train data Loaded!!!"
        return

    #Load Question data - Gets the question + sequence data
    def loadQuestionData(self):
        if self.isQuestionLoaded==False:
            print "Questions are Loading..."
            self.questionY = Config().getQuestionsData()
            self.isQuestionLoaded=True
        print "Questions Loaded!!!"
        return

    # Load CrossValidation data if not loaded already
    def loadCrossValidationData(self):
        if self.isCrossValidationLoaded==False:
            print "CrossValidation data is Loading..."
            resultsSet = Config().getCrossValidationFeatureData()
            for row in resultsSet:
                self.crossX.append(row[:-1])
                self.crossY.append(int(row[-1:][0]))
            self.isCrossValidationLoaded=True
        print "CrossValidation data Loaded!!!"
        return
    
    # Generates SVM for the Train data
    def generateSVM(self):
        #svmfile = 'svm_persistent'
        #svmfile = 'svm_persistent_ovr'
        svmfile = 'svm_persistent_ovr2'
        #svmkey = 'svm_polygon'
        #svmkey = 'svm_fft'
        svmkey = 'svm_ovr'
        #svmkey = 'svm_sigmoid'
        d = shelve.open(svmfile)        
        if self.isSVMPersisted==True:
            print "Generating SVM from the persistent Store..."
            self.clf = d[svmkey]
            print "SVM Generated!!!"
        elif self.isSVMGenerated==False:
            print "Generating SVM for the input data..."
            print self.clf.fit(self.trainX, self.trainY)
            print "SVM Generated!!!"
            print "Persisting SVM into hard disk!!!"
            d[svmkey] = self.clf
            print "SVM Persisted!!!"
        '''
        print "Support Vectors: "
        print self.clf.support_vectors_
        print "Support Vector Indices: "
        print self.clf.support_
        print "Number of Support Vectors for each of the classes: "
        print self.clf.n_support_
        '''
        self.isSVMGenerated=True        
        return

    # Generates RF for the Train data
    def generateRF(self):
        rffile = 'rf_persistent'
        rfkey = 'rf_key'
        d = shelve.open(rffile)
        if self.isRFPersisted==True:
            print "Generating RF from the persistent store..."
            self.rf = RandomForestClassifier(d[rfkey])
        elif self.isRFGenerated==False:
            print "Generating RF for the input data..."
            print self.rf.fit(self.trainX, self.trainY)
            # d[rfkey] = str(self.rf)
        print "RF Generated!!!"
        return
    
    # cross validates input training data
    def crossValidateTrainingData(self):
        # loading training and crossvalidation data as required
        print "Cross Validation under Progress..."
        self.loadTrainData()
        self.loadCrossValidationData()
        self.generateSVM()
        # self.generateRF()
        # Testing the system
        count=0
        trials = 10
        for index in range(len(self.crossX)):
            predictedOutput = self.clf.predict(self.crossX[index])
            #probabilities = self.clf.predict_proba(self.crossX[index])
            #predictedDecisions = self.clf.decision_function(self.crossX[index])
            #print predictedDecisions
            #print predictedDecisions.scores
            #self.clf.score(self.crossX[index], 
            #predictedOutput = self.rf.predict(self.crossX[index])
            predictedValue = int(predictedOutput[0])
            actualValue = int(self.crossY[index])
            if predictedValue==actualValue :
                count += 1
            #trials -= 1
            #if trials<=0:
            #    break
        print float(count)*100/len(self.crossX), "% Success"
        print "Cross Validation Completed!!!"
        return

    # FFT With Graph
    def fft_train(self):
        # trainData = Config().getOriginalTrainData()
        # 100 sec signal - 5 samples/sec - 0 starting timeslot
        noTrials = 5
        self.trainX = []
        self.trainY = []
        for deviceRow in Config().getDistinctDevices(): 
            noSamplesPerDevice = 1000
            samplesets = 10
            deviceNumber = deviceRow[0]
            while samplesets>0:
                trainData = Config().getOriginalTrainDataForDevice(deviceNumber)
                trainDataLength = len(trainData)
                t = scipy.linspace(0, noSamplesPerDevice/5, noSamplesPerDevice)
                
                limitOffset = random.randint(0, trainDataLength-noSamplesPerDevice)
                trainDataSample = trainData[limitOffset : limitOffset+noSamplesPerDevice]
                signal = np.asarray(trainDataSample)
                FFT = abs(scipy.fft(signal))
             
                fftList = FFT.tolist()
                fftList = [int(i[0]) for i in fftList]
                self.trainX.append(fftList)
                self.trainY.append(int(deviceNumber))
                
                freqs = scipy.fftpack.fftfreq(len(signal), t[1]-t[0])
                # print deviceNumber
                samplesets -= 1
                '''
                noTrials -= 1
                if noTrials==0:
                    break
                # end of while
                '''
            print deviceNumber, ": Sampling complete!!!"
            '''
            if noTrials==0:
                break
            # end of for
            '''
        print len(self.trainX)
        print len(self.trainY)
        # self.generateSVM()
        self.generateRF()
        '''
        pylab.subplot(211)
        pylab.plot(t, signal)
        pylab.subplot(212)
        pylab.plot(freqs,20*scipy.log10(FFT),'x')
        pylab.show()
        '''
        return
    
    def fft_crossvalidate(self):
        # trainData = Config().getOriginalTrainData()
        # 100 sec signal - 5 samples/sec - 0 starting timeslot
        self.generateSVM()
        
        noSamplesPerDevice = 1000
        samplesets = 10
        
        count = 0
        arrDevices = Config().getDistinctDevices()
        for deviceRow in arrDevices: 
            deviceNumber = deviceRow[0]
            tempSamplesets = samplesets
            while tempSamplesets>0:
                trainData = Config().getOriginalTrainDataForDevice(deviceNumber)
                trainDataLength = len(trainData)
                t = scipy.linspace(0, noSamplesPerDevice/5, noSamplesPerDevice)
                
                limitOffset = random.randint(0, trainDataLength-noSamplesPerDevice)
                trainDataSample = trainData[limitOffset : limitOffset+noSamplesPerDevice]
                signal = np.asarray(trainDataSample)

                FFT = abs(scipy.fft(signal))
                fftList = FFT.tolist()
                fftList = [int(i[0]) for i in fftList]
                freqs = scipy.fftpack.fftfreq(len(signal), t[1]-t[0])
                
                # predictedOutput = self.clf.predict(fftList)
                predictedOutput = self.rf.predict(fftList)
                if(int(deviceNumber) == int(predictedOutput)):
                    count += 1
                # print deviceNumber
                tempSamplesets -= 1   
            print deviceNumber, ": crossvalidation complete!!!"
        print float(count)*100.0/(len(arrDevices)*samplesets), "% Success!!!"
        return
    
    def fft_test(self):
        # trainData = Config().getOriginalTrainData()
        # 100 sec signal - 5 samples/sec - 0 starting timeslot
        noTrials = 5
        self.trainX = []
        self.trainY = []
        for deviceRow in Config().getDistinctDevices(): 
            noSamplesPerDevice = 1000
            samplesets = 10
            deviceNumber = deviceRow[0]
            while samplesets>0:
                trainData = Config().getOriginalTrainDataForDevice(deviceNumber)
                trainDataLength = len(trainData)
                t = scipy.linspace(0, noSamplesPerDevice/5, noSamplesPerDevice)
                
                limitOffset = random.randint(0, trainDataLength-noSamplesPerDevice)
                trainDataSample = trainData[limitOffset : limitOffset+noSamplesPerDevice]
                signal = np.asarray(trainDataSample)
                FFT = abs(scipy.fft(signal))
             
                fftList = FFT.tolist()
                fftList = [int(i[0]) for i in fftList]
                self.trainX.append(fftList)
                self.trainY.append(int(deviceNumber))
                
                freqs = scipy.fftpack.fftfreq(len(signal), t[1]-t[0])
                # print deviceNumber
                samplesets -= 1
                '''
                noTrials -= 1
                if noTrials==0:
                    break
                # end of while
                '''
            print deviceNumber, ": Sampling complete!!!"
            '''
            if noTrials==0:
                break
            # end of for
            '''
        print len(self.trainX)
        print len(self.trainY)
        self.generateSVM()
        return

    
    # Classification - A
    def classify(self):
        print "Classification under progress!!!"
        # Loading training, testing and question data
        self.loadTrainData()
        self.loadTestData()
        self.loadQuestionData()
        # self.generateSVM()
        self.generateRF()
        # Testing the system

        outputFile = open('output.csv', 'w')
        outputFileWriter = csv.writer(outputFile, delimiter=',')
        outputFileWriter.writerow(["QuestionId", "IsTrue"])
        testY = {}
        for index in range(len(self.testX)):
            # predictedOutput = self.clf.predict(self.testX[index])
            predictedOutput = self.rf.predict(self.testX[index])
            predictedValue = int(predictedOutput[0])
            testY[ self.testS[index] ] = predictedValue         
        for index in range(len(self.questionY)):
            sValue = int(self.questionY[index][1])
            actualValue = int(self.questionY[index][2])
            predictedValue = testY[sValue]
            if actualValue == predictedValue:
                outcome=1
            else:
                outcome=0
            outputRow = []
            outputRow.append(self.questionY[index][0])
            outputRow.append(outcome)
            outputFileWriter.writerow(outputRow)
        print "Classification Completed!!!"
        return 


# Setting up the project - Configuration
# Database and stuff

#conf = Config()
#conf.loadTrainData("inputs/accelerometer_train.csv");
#conf.loadTestData("inputs/accelerometer_test.csv");
#conf.comparePrecision()
#conf.loadFeatureData()
#conf.loadCrossValidationFeatureData()
#conf.loadTestFeatureData()


# Accelerometer Biometric - Business Logic
# Function calls for initialization, classification and testing

accBiometric = AccBiometric()
#accBiometric.fft_train()
#accBiometric.fft_test()
#accBiometric.fft_crossvalidate()
accBiometric.crossValidateTrainingData()
#accBiometric.classify()


