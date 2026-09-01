#!/usr/bin/env python

import math
from clusterKMeans import Point

class LNormalizer:
    featureData = []
    
    # computed
    mins = []
    maxs = []
    refs = []
    
    def __init__(self,featureData):
        self.featureData = featureData
    
    def normalizeFeatures(self):
        self.points2Arr()
        transposedFeatures = self.transposeMatrix(self.featureData)
        self.maxs,self.mins = self.getStastics(transposedFeatures)
        transNormedFeatures = self.normalize(transposedFeatures)
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
        maxs = []
        mins = [] 
        for line in matrix:
            maxs.append(max(line))
            mins.append(min(line))
        return maxs,mins
    
    def normalize(self,matrix):
        index = 0
        normMatrix = []
        #print("Maxis : "+str(self.maxs))
        #print("Minis : "+str(self.mins))
        for line in matrix:
            newLine = []
            maxi = self.maxs[index]
            mini = self.mins[index]
            for value in line:
                if (maxi == mini):
                    newLine.append(0.0)
                else:
                    newLine.append(((value-mini) /(maxi-mini)))
            normMatrix.append(newLine)
            index += 1
        return normMatrix
    
    def normalizeVector(self,vector):
        normedVector = []
        index = 0
        mini = self.mins
        maxi = self.maxs
        for value in vector:
            if (mini[index] != maxi[index]):
                normedVector.append((float(value)-mini[index])/(maxi[index] - mini[index]))
            else:
                normedVector.append(0.0)
            index += 1
        return normedVector