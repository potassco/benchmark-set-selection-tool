#!/usr/bin/env python

import math

class ZNormalizer:
    featureData = []
    
    # computed
    means = []
    vars = []
    refs = []
    
    def __init__(self,featureData):
        self.featureData = featureData
    
    def normalizeFeatures(self):
        self.points2Arr()
        transposedFeatures = self.transposeMatrix(self.featureData)
        self.means,self.vars = self.getStastics(transposedFeatures)
        transNormedFeatures = self.normalize(transposedFeatures)
        means_int,vars_int = self.getStastics(transNormedFeatures)
        self.featureData = self.transposeMatrix(transNormedFeatures)
        self.arr2Points()
        return self.featureData
    
    def points2Arr(self):
        arr = []
        for p in self.featureData:
            arr.append(p.coords)
            self.refs.append(p.reference)
        self.featureData = arr
        return True

    def arr2Points(self):
        from clusterKMeans import Point
        points = []
        for a in self.featureData:
            ref = self.refs.pop(0)
            points.append(Point(a,ref))
        self.featureData = points
    
    def transposeMatrix(self,matrix):
        transposedMatrix = []
        for row in matrix:
            index = 0
            for value in row:
                try:
                    transposedMatrix[index].append(value)
                except:
                    transposedMatrix.append([value])
                index += 1
        return transposedMatrix
    
    def getStastics(self,matrix):
        means = []
        vars = [] 
        for line in matrix:
            sum = 0
            var = 0
            for value in line:
                sum += value
                var += value * value
            n = len(line)
            mean = sum/n
            means.append(mean)
            var = var/n - mean*mean
            if (var > 0.001):
                vars.append(var)
            else:
                vars.append(0.001)
        return means,vars
    
    def normalize(self,matrix):
        index = 0
        normMatrix = []
        #print("Means : "+str(self.means))
        #print("Vars : "+str(self.vars))
        for line in matrix:
            newLine = []
            mean = self.means[index]
            std = math.sqrt(self.vars[index])
            for value in line:
                newLine.append((float(value)-mean)/std)
            normMatrix.append(newLine)
            index += 1
        return normMatrix
    
    def normalizeVector(self,vector):
        normedVector = []
        index = 0
        means = self.means
        vars = self.vars
        for value in vector:
            normedVector.append((float(value)-means[index])/math.sqrt(vars[index]))
            index += 1
        return normedVector