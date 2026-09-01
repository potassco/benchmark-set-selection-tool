#!/usr/bin/env python
# Author: pandoricweb http://pandoricweb.tumblr.com/post/8646701677/python-implementation-of-the-k-means-clustering
# modified by: Marius Schneider
# Date 30th May 2012

import sys, math, random
import argparse
 
import zNormalizer
#import lNormalizer 

class Point:
    def __init__(self, coords, reference=None):
        self.coords = coords
        self.n = len(coords)
        self.reference = reference
    def __repr__(self):
        #return str(self.coords)
        return str(self.reference)

class Cluster:
    def __init__(self, points):
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        self.points = points
        self.n = points[0].n
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")
        self.centroid = self.calculateCentroid()
    def __repr__(self):
        return str(self.points)
    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        return getDistance(old_centroid, self.centroid)
    def calculateCentroid(self):
        if (self.points != []):
            reduce_coord = lambda i:reduce(lambda x,p : x + p.coords[i],self.points,0.0)
            centroid_coords = [reduce_coord(i)/len(self.points) for i in range(self.n)]
            return Point(centroid_coords)
        else:
            num_points, dim, lower, upper = 1, self.n, -1, 1
            point = map( lambda i: makeRandomPoint(dim, lower, upper), range(num_points) )
            return point[0]
    def getQuality(self):
        dists = 0
        if (len(self.points) == 0):
            return 0
        for p in self.points:
            dists += getDistance(p, self.centroid)
        return dists/len(self.points)

def kmeans(points, k, cutoff, maxIts):
    initial = random.sample(points, k)
    clusters = [Cluster([p]) for p in initial]
    iterations = 0
    while iterations <= maxIts:
        iterations += 1
        lists = [ [] for c in clusters]
        for p in points:
            smallest_distance = getDistance(p,clusters[0].centroid)
            index = 0
            for i in range(len(clusters[1:])):
                distance = getDistance(p, clusters[i+1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i+1
            lists[index].append(p)
        biggest_shift = 0.0
        for i in range(len(clusters)):
            shift = clusters[i].update(lists[i])
            biggest_shift = max(biggest_shift, shift)
        if biggest_shift < cutoff: 
            break
    return clusters

def getDistance(a, b):
    if a.n != b.n: raise Exception("ILLEGAL: non comparable points")
    ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)

def makeRandomPoint(n, lower, upper):
    return Point([random.uniform(lower, upper) for i in range(n)])

def cluster(points,k):
    #num_points, dim, k, cutoff, lower, upper = 10, 4, 3, 0.5, 0, 200
    #points = map( lambda i: makeRandomPoint(dim, lower, upper), range(num_points) )
    cutoff = 2
    maxIterations = 1000
    clusters = kmeans(points, k, cutoff,maxIterations)

#    for i,c in enumerate(clusters): 
#        for p in c.points:
#            print " Cluster: ",i,"\t Point :", p 

    if(__debug__):
        print("Quality: ")
    quals = 0
    for i,c in enumerate(clusters):
        qual = c.getQuality()
#        print("Cluster "+str(i)+" : "+str(qual) +"\t Mass : "+str(len(c.points)))
        quals += qual*len(c.points)
    if(__debug__):        
        print("Overall : "+str(quals/len(points)))
    return clusters,quals/len(points)
        
def parseFeatures(featureFile):
    points = {}
    try:
        fh = open(featureFile, "r")
        nFeats = -1
        for line in fh:
            line = line.split(",")
            values = []
            instName = line.pop(0)
            for value in line:
                try:
                    values.append(float(value))
                except:
                    pass
            if (values != [] and (nFeats == -1 or len(values) == nFeats)):    # filter empty lines and wrong diments 
                point = Point(values,instName)
                points[instName] = (point)
                entry = points.get(instName)
                if (entry != None):
                    sys.stderr.write("Warning Overwrite: duplication of feature data for " + str(instName) + "\n")
                if (nFeats == -1):
                    nFeats = len(values)
            else:
                sys.stderr.write("WARNING: "+str(instName)+" has the wrong number of dimensions\n")
                sys.stderr.write(str(values)+"\n")
    except:
         traceback.print_exc(file=sys.stderr)
         sys.stderr.flush()
         return False
    if(__debug__):
        print(">>>Feature Data:<<<")
        print(points)
        print("Reading Features was sucessful!")
    return points

def executeClustering(points,reps,k):
    bestClusters = None
    bestQual = 100000000
    for i in range(0,reps):
        clusters,qual = cluster(points,k)
        if (qual < bestQual):
            bestClusters = clusters
            bestQual = qual
    print("Best Quality: "+str(bestQual))
    for i,c in enumerate(bestClusters):
        qual = c.getQuality()
        print("Cluster "+str(i)+" : "+str(qual) +"\t Mass : "+str(len(c.points)))
    #sizeOfData(bestClusters)
    return bestClusters

def sizeOfData(clusterList):
    i = 0
    for clu in clusterList:
            for inst in clu.points:
                i += 1
    print("SIZE: "+str(i))
    
def getNParts(points,folds):
    l = len(points)
    index = 0
    pointsParts = []
    internalList = []
    partIndex = 1
    threshold = l/folds
    partIndex = 1
    pointsCopy = points[:]
    while pointsCopy != []:
        randIndex = random.randint(0,len(pointsCopy)-1)
        point = pointsCopy.pop(randIndex)
        internalList.append(point)
        if (index >= threshold):
            pointsParts.append(internalList)
            internalList = []
            l = l - index
            threshold = l/ (folds-partIndex)
            partIndex += 1
            index = 0
        index += 1
    pointsParts.append(internalList)
    return pointsParts

def joinFolds(list,exclude):
    joinedList = []
    index = 0
    for l in list:
        if (index != exclude):
            joinedList.extend(l)
        index += 1
    return joinedList

def doCluster(seed,feature,reps,clus,findK,readIn):
    random.seed(seed)
    if (readIn == True):
        points = parseFeatures(feature).values()
    else:
        points = []
        for inst,feats in feature.items():
            points.append(Point(feats,inst))
 #   norma = lNormalizer.LNormalizer(points)
    norma = zNormalizer.ZNormalizer(points)
    points = norma.normalizeFeatures()
    
    if (findK > 0):
        pointsParts = getNParts(points,findK)
        
        #iterate over number of clusters
        allMinDists = []
        bestK = -1
        for k in range(2,int(math.sqrt(len(points)/2))):
            # compute cross validation 
            sumDist = 0
            for i in range(0,findK):
                #training
                training = joinFolds(pointsParts,i)
                clusters = executeClustering(training,max([int(reps/10),1]),k)
                #validation on test set
                test = pointsParts[i]
                for datum in test:
                    listDist = []
                    for cluster in clusters:
                        centroid = cluster.centroid
                        listDist.append(getDistance(datum, centroid))
                    sumDist += min(listDist)
            allMinDists.append(sumDist)
            if (len(allMinDists) > 1 and sumDist > allMinDists[-2]):
                break
                
        bestK = allMinDists.index(min(allMinDists))+2
        print("Dists: "+str(allMinDists))        
        print("Best K: "+str(bestK))
        clusters = executeClustering(points,reps,bestK)
    else:
        clusters = executeClustering(points,reps,clus)     
        
    return clusters

    
if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser()
    req_group = parser.add_argument_group("Options")
    req_group.add_argument('--csv', dest='csv', action='store', required=True,  help='csv file with features')
    req_group.add_argument('--k', dest='k', action='store', required=False, type=int, help='number of clusters')
    req_group.add_argument('--r', dest='r', action='store', required=True, type=int, help='repetitions')
    req_group.add_argument('--s', dest='s', action='store', default=1243, required=True, type=int, help='random seed')
    req_group.add_argument('--findK', dest='findK', action='store', default=-1, type=int, help='use cross fold validation to find the number of clusters (k)')
    
    
    args = parser.parse_args()     
    
    doCluster(args.s,args.csv,args.r,args.k,args.findK,True)
    
    
    
        
