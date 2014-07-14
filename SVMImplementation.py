# -*- coding: utf-8 -*-
# !/ usr / bin / env python

#Our implementation of SVM
from cvxopt import *
from cvxopt . solvers import qp
#from model import Model
from cvxopt import matrix
from math import exp
from math import tanh
import numpy
import MySQLdb


class DBUtil:
    def getDBConnection(self):
        # Socket location of Mysql-TukoDB: /tmp/mysql.sock
        # Port Number: 3306
        unix_socket = '/tmp/mysql.sock'
        db = MySQLdb.connect(host="localhost", user="root",port=3306, passwd="ramsri", db="digitrecognizer",  read_default_file="/etc/my.cnf")
        return db
    def printCreateTable(self):
        print """create table digitrec
                (
                """
        i=0
        print "label INT,"
        while i<784:
            var = "pixel"+str(i)
            print var, "INT,"
            i+=1
        print ")"
        
    def getTraindata(self):
        # Creating database
        db = self.getDBConnection()
        cursor = db.cursor()
        #SQL query to INSERT a record into the table trainData.
        cursor.execute('''SELECT * FROM digitrec where label=9 limit 400 UNION (SELECT * FROM digitrec where label=1 limit 400)''')
        results = cursor.fetchall();
        num_rows = cursor.rowcount
        '''
        x = map(list, list(results))
        x = sum(x, [])
        D = numpy.fromiter(iterable=x, dtype=float, count=-1)
        D = D.reshape(num_rows, -1)
        '''
        return results

    def getTestdata(self):
        # Creating database
        db = self.getDBConnection()
        cursor = db.cursor()
        #SQL query to INSERT a record into the table trainData.
        cursor.execute('''SELECT * FROM digitrec where label=9 limit 100 offset 400 UNION (SELECT * FROM digitrec where label=1 limit 100 offset 400)''')
        results = cursor.fetchall();
        num_rows = cursor.rowcount
        '''
        x = map(list, list(results))
        x = sum(x, [])
        D = numpy.fromiter(iterable=x, dtype=float, count=-1)
        D = D.reshape(num_rows, -1)
        '''
        return results

        
class SupportVectorMachine :
    def __init__ ( self , kf ):
        self.__kernelFunction = kf
        
    def data ( self , labels , features ):
        #self.__labels = numpy.array([1, -1, -1, 1, -1])
        #self.__features = numpy.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        #[[1], [2], [3], [4], [5]]
        #self.__labels = numpy.array([1, 1, 1, -1, -1])
        #self.__features = numpy.array([[-1, 0], [-2, 0], [-3, 0], [0, 1], [0, 2]])
        #self.__labels = numpy.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        #self.__features = numpy.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16],
        #                            [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [26, 26], [31, 31], [32, 32], [33, 33], [34, 34], [35, 35], [36, 36]])
        #self.__labels = numpy.array([1, 1, -1, -1, -1])
        #self.__features = numpy.array([[-10, 1], [-8, 2], [-3, -3], [-4, -4], [-5, -5]])
        #self.__labels = numpy.array([1, 1, -1, -1, -1])
        #self.__features = numpy.array([[-10, 1], [-1, 0], [1, 0], [2, 0], [2, 2]])
        #self.__features = numpy.array()
        #self.__labels = numpy.array()
        features = []
        labels = []
        for row in DBUtil().getTraindata():
            features.append(list(row[1:]))
            if int(row[0]) == 1:
                labels.append(1)
            else:
                labels.append(-1)
        #print type(features[0])
        #print type(labels[0])
        #print len(features)
        #print len(labels)
        #print features
        #print labels
        self.__features = numpy.asarray(features)
        self.__labels = numpy.asarray(labels)
        self.__rows = 800;
        self.__nofeatures = 784;
        return

    def dotproduct(self, x1, x2):
        mysum = 0.0;
        for i in range(self.__nofeatures):
            mysum += x1[i]*x2[i]
        return mysum
    
    def optimize ( self ):
        #Initialize Variables
        H = matrix (0.0 ,( self . __rows , self . __rows ))
        G = matrix (0.0 ,( self . __rows , self . __rows ))
        q = matrix ( -1. ,( self . __rows ,1))
        h = matrix (0. ,( self . __rows ,1))
        # Short notation
        #kf = self.__kernelFunction
        #kf = KernelMachine().linear
        y = self.__labels
        x = self.__features
        # Compute H
        for idx in xrange ( self . __rows ):
            for idy in xrange ( self . __rows ):
                #print self.dotproduct ( x [ idx ,:] , x [ idy ,:])
                H[ idx , idy ] = y[idx ]*y[idy]*float(self.dotproduct ( x [ idx ,:] , x [ idy ,:]))
                #y[idx ]*y[idy]*self.dotproduct ( x [ idx ,:] , x [ idy ,:])
        #print H
        # Compute G
        for idx in xrange ( self . __rows ):
            G [ idx , idx ] = -1.
        # Solve
        print "Solving"
        self.__solver = qp (H , q , G , h )
        print "Solved"
    
    def model ( self , km ):
        # Filter alpha ’s
        alpha = list (self . __solver ['x'])
        '''
        for i in xrange (self.__rows):
            alpha[i] = -alpha[i]
        '''
        #print alpha
        #return Model (km, self.__kernelFunction, labels , alphas , svs )
        self.w = []
        for i in range(0, self.__nofeatures):
            self.w.append(0)
        x = self.__features;
        y = self.__labels;
        # calculating w from the given constraints
        for idx in xrange (self.__rows):
            item = alpha[idx]*self.__labels[idx]*self.__features[idx]
            #print alpha[idx]*self.__labels[idx]*self.__features[idx]
            for i in xrange (self.__nofeatures):
                self.w[i] += item[i]
        # calculating b from the constraints
        '''
        for i in xrange (self.__nofeatures):
            self.w[i] = -self.w[i]
        '''
        minValue = 10000000
        maxValue = -1000000
        for i in xrange (self.__rows):
            value = self.dotproduct( x[i], self.w)
            print x[i], ": ", value, " - ", y[i]
            if y[i] == 1:
                if(maxValue<value):
                    maxValue = value
            else:
                if(minValue>value):
                    minValue = value
                    
        self.b = -(float(maxValue) + minValue) / 2.0;
        print "W: ", self.w
        print "B: ", self.b
    
    def test ( self):
        i=0
        count=0
        for row in DBUtil().getTestdata():
            x = row[1:]
            if (self.dotproduct(x, self.w)+self.b)>0:
                print "Device 1"
                if i>=100:
                    count+=1
            else:
                print "Device 4"
                if i<100:
                    count+=1
            i+=1
        print float(count)*100/200.0, "% success!!!";
'''
class KernelMachine :    
    def __init__ ( self ):
        self . __linear_const = 1
        self . __params = {}
        
    def param ( self , key , value ):
        self . __params [ key ] = value
        
    def linear ( self ,x , y ):
        return ( x * y  + self . __params [ ' linear_constant ' ])[0]
    
    def sigmoid ( self ,x , y ):
        g = self . __params [ ' gamma ']
        k = self . __params [ 'k ']
        return ( g * x * y . T - k )[0]

    def polynomial ( self ,x , y ):
        degree = self . __params [ ' degree ']
        return (( x * y . T + 1)[0])** degree

    def radialbasis ( self ,x , y ):
        g = self . __params [ ' gamma ']
        z = x - y
        return exp ( - ( z * z . T )[0] / (2 * g **2) )

    def wradialbasis ( self ,x , y ):
        g = self . __params [ ' gamma ']
        w = self . __params [ ' kw ']
        o = 0.0
        for idx in xrange ( len ( w )):
            o += w [ idx ] * (( x [ idx ] - y [ idx ])*( x [ idx ] - y [ idx ]))
        return exp ( - g * o )
'''
#KernelMachine().linear

mySvm = SupportVectorMachine('linear')
mySvm.data(1, 2)
mySvm.optimize()
mySvm.model("km")
mySvm.test()

#DBUtil().printCreateTable()
#DBUtil().getTraindata()
