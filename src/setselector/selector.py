#!/usr/bin/env python
#author: Marius Schneider

#standard imports

import os
import sys
import argparse
import traceback
import math
import copy
import operator

#own imports

import clusterKMeans

class Selector:
    '''
        selects a representative set which is good for benchmarking
    '''   
       
    printErrs = True
    solvers = []
    __feature_data_dic = {}
    _runtime_data_dic = {}
    runtimeData = []
    featureData = []
    __clusters = {}
    clustersStr = []
    cutoff = 5000
    __n_clusters = 10
       
    def __init__(self,cutoff):
        ''' constructor '''
        self.cutoff = cutoff

        #self.printErrs = True
        #self.solvers = []
        #self.__feature_data_dic = {}
        #self._runtime_data_dic = {}
        #self.runtimeData = []
        #self.featureData = []
        #self.__clusters = {}
        #self.clustersStr = []
        #self.cutoff = 5000
        #self.__n_clusters = 10
        
    
    def parse_features(self, feature_file):
        '''
            parse csv feature file 
            Args:
                feature_file: csv file with features (first col instance name)
            Returns:
                True if successful
                False else
        '''
        try:
            fh = open(feature_file,"r")
            for line in fh:
                line = line.split(",")
                values = []
                instName = line.pop(0)
                for value in line:
                    try:
                        values.append(max([float(value),-1.0])) # minimal value -1; negative values are missing values (e.g. -512 satzilla)
                    except:
                        pass
                if (values != []):    # filter empty lines 
                    entry = self.__feature_data_dic.get(instName)
                    self.__feature_data_dic[instName] = (values)
                    if (entry != None):
                        sys.stderr.write("Warning Overwrite: duplication of feature data for "+str(instName)+"\n")
        except:
            if (self.printErrs):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            return False
        if(__debug__):
            print(">>>Feature Data:<<<")
            print(self.__feature_data_dic)
        print("Reading Features was sucessful!")
        return True
    
    def parse_runtimes(self, runtimefile):
        '''
            parse csv runtime file 
            Args:
                feature_file: csv file with runtimes (first col instance name)
            Returns:
                True if successful
                False else
        '''
        try:
            fh = open(runtimefile,"r")
            header = fh.readline()
            self.solvers = header.replace("\n","").split(",")
            for line in fh:
                line = line.split(",")
                values = []
                instName = line.pop(0)
                for value in line:
                    try:
                        value = min(self.cutoff, float(value))
                        values.append(value)
                    except ValueError:
                        pass
                if (values != []): # filter empty lines
                    entry = self._runtime_data_dic.get(instName)
                    self._runtime_data_dic[instName] = (values)
                    if (entry != None):
                        sys.stderr.write("Warning Overwrite: duplication of runtime data for "+str(instName)+"\n")
        except:
            if (self.printErrs):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            return False
        if(__debug__):
            print(">>>Runtime Data:<<<")
            print(self._runtime_data_dic)
        print("Reading Runtimes was sucessful!")
        return True
    
    def randomTestTrainingSplit(self):
        '''
            splits instances in training and test instances
            all test instances are remove from the data structures and printed
        '''
        import random
        random.seed(1234)
        instances = self._runtime_data_dic.keys()
        n = len(instances)
        print(">>> Test Instances (remaining instances are used as training instances): ")
        for index in range(0,n/2):
            s_instance = random.randint(0,n-index)
            instance = instances.pop(s_instance)
            print("%s" %(instance))
            self._runtime_data_dic.pop(instance)
            try:
                self.__feature_data_dic.pop(instance)
            except KeyError:
                pass
    
    def joinTimesFeatures(self):
        '''
            joins the features and runtimes (intersection of instance names)
            Sets:
                _runtime_data_dic
                __feature_data_dic
        '''
        available = 0
        runtimeDataDicLocal = {}
        featureDataDicLocal = {}
        length_feats = -1
        for inst,times in self._runtime_data_dic.items():
            if (sum(times) == len(times)*(self.cutoff)): # filter instances with only timeouts
                #mark too hard instances with cluster -1
                self.__clusters[str(inst)] = "h"
                continue
            features = self.__feature_data_dic.get(inst)    # assumption: first feature has correct number of features
            if (length_feats == -1):
                length_feats = len(features)
            if (features == None):
                sys.stderr.write("Warning: there are runtime data but no features available for "+str(inst)+"\n")
                self.__clusters[str(inst)] = "f" # mark instances with failed feature extraction with -3
                continue
            if (len(features) != length_feats):
                sys.stderr.write("Warning: Invalid number of features for "+str(inst)+" : "+str(len(features))+"\n")
                self.__clusters[str(inst)] = "f"
                continue
            if (sum(features) == 0.0): # error output of feature extraction
                continue
            if (math.isnan(sum(features)) or math.isinf(sum(features))):
                continue
                sys.stderr.write("Warning: feature data include NAN or INF in "+str(inst)+"\n")
                self.__clusters[str(inst)] = "f"
            # else everything is ok
            featureDataDicLocal[inst] = features
            runtimeDataDicLocal[inst] = times
            self.runtimeData.append(times)
            self.featureData.append(features)
            available += 1
        self._runtime_data_dic = runtimeDataDicLocal
        self.__feature_data_dic = featureDataDicLocal
      #  self._runtime_data_dic.clear()
      #  self.__feature_data_dic.clear()
        print(">>> Available Data :"+str(available))
    
    def clustering(self,reps):
        '''
            cluster instances (kmean) based on distance in feature space
            Args:
                reps: repetitions of clustering (start sensitive)
            Sets:
                __clusters
                __n_clusters
        '''
        clusterList = clusterKMeans.doCluster(123456,self.__feature_data_dic,reps,-1,10,False) #seed,feature,reps,clus,findK,readIn
        #self.__clusters = {}
        clusterIndex = 0
        
        for clu in clusterList:
            for inst in clu.points:
                self.__clusters[str(inst)] = clusterIndex
            clusterIndex += 1
        self.__n_clusters = clusterIndex + 1
        
        for inst, cluster in self.__clusters.items():
            print("%s,%s" % (inst, str(cluster)))
        print("")
        print("Data in Clusters: "+str(len(self.__clusters)))

    def select(self, n, frac, agg, dist):
        '''
            selects instances based on gaussian or uniform distribution and feature clusters
            Args:
                n: number of instances to select
                frac: maximal fraction of representation of one cluster
                agg: how to aggregate instance hardness (ind, avg or min)
                dist: select distribution (gauss or uni)
            Returns:
                samples: list of instances
        '''
        import random
        random.seed(1234)
        samples = []
        sampled = 0
        sorted_inst = self.sortInst(agg)
        print("Number of Clusters "+str(self.__n_clusters))
        cluster_reps = self.__n_clusters*[0] # self.get_vector(0,self.__n_clusters)
        mean,variance = self.get_runtime_statistics(sorted_inst)
        print("Mean: "+str(mean)+"\t Variance: "+str(variance))
        while (sampled < n and sorted_inst != []):
            if (dist == "gauss"):
                sample = random.gauss(mean,math.sqrt(variance))
            if (dist == "uni"):
                sample = random.random() * float(self.cutoff)
            if (dist == "exp"):
                sample = random.expovariate(1/mean)
            if (dist == "log"):
                sample = random.lognormvariate(math.log(mean),math.log(math.sqrt(variance)))
            #print(sample)
            #print(len(sorted_inst))
            (inst,aggValue) = self.find_nearest(sample,sorted_inst)
            inst = inst.split(",")[0]
            print((inst,aggValue))
            #print(sorted_inst)
            cluster = self.__clusters[inst]
            print("Cluster: "+str(cluster))
            if ((not self.is_overrepresented(cluster,cluster_reps,frac,n)) and samples.count(inst) == 0):
                samples.append(inst)
                sampled += 1
                print("ACCEPTED")
            else:
                print("REJECTED")
        print("Remaining Instances: "+str(len(sorted_inst)))
        self.__samples = samples
        return samples
    
    def removeTooEasy(self,cutoff,threshold):
        '''
            remove too easy instances (< threshold*cutoff)
            Args:
                cutoff: of measured runtime
                threshold: fraction of cutoff
            Modifies:
                _runtime_data_dic
                __feature_data_dic
        '''
        removeable = []
        for inst,vec in self._runtime_data_dic.items():
            #avg = float(sum(vec))/len(vec)
            maxi = max(vec)
            if (max(vec) < cutoff and maxi < threshold*cutoff):
                removeable.append(inst)
                # mark too easy instances with cluster -2
                self.__clusters[inst] = "e"
        for rem in removeable:
            self._runtime_data_dic.pop(rem)
            self.__feature_data_dic.pop(rem)
        print("Remaining Instances after Easy Filtering: "+str(len(self._runtime_data_dic)))

    def sortInst(self, agg):
        '''
            sort instances by hardness aggregation
            Args:
                agg: how to aggregate instance hardness (ind, avg or min)
            Returns:
                sorted_tuples: (instances,instance hardness)
        '''
        instAvgDic = {}
        for inst,vec in self._runtime_data_dic.items():
            if (agg == "ind"):
                index = 1
                for v in vec:
                    instAvgDic[inst+","+str(index)] = v 
                    index += 1
                continue
            if (agg == "avg"):
                aggvec = float(sum(vec))/len(vec)
            if (agg == "min"):
                aggvec = float(min(vec))
            instAvgDic[inst] = aggvec
        sorted_tuples = sorted(instAvgDic.iteritems(), key=operator.itemgetter(1))
       # print(sorted_tuples)
        return sorted_tuples

    def find_nearest(self, value, sorted_tuples):
        '''
            given an instance hardness value, find nearest instance
            Args:
                value: float
                sorted_tuples: (instance, hardness)
            Returns:
                tuple: nearest tuple to value
        '''
        lastInst = None
        lastAvg = 0
        index = 0
        for (inst,avg) in sorted_tuples:
            if (avg > value and lastInst == None):
                sorted_tuples.pop(index)
                return inst,avg
            if (avg > value and lastInst != None):
                if (avg-value < lastAvg - value):
                    sorted_tuples.pop(index)
                    return inst,avg
                else:
                    sorted_tuples.pop(index-1)
                    return lastInst,lastAvg
            index += 1
            lastAvg = avg
            lastInst = inst
            
        last = len(sorted_tuples)-1
        tuple = sorted_tuples.pop(last)
        return tuple
            
    def get_runtime_statistics(self,sorted_tuples):
        '''
            get mean and variance of instance hardnesses
            Args:
                sorted_tuples: (instance, hardness)
            Returns:
                mean, variance
        '''
        sum = 0
        sumsqr = 0
        for (inst,avg) in sorted_tuples:
            sum += avg
            sumsqr += avg*avg
        n = len(sorted_tuples)
        mean = sum / n
        variance = sumsqr / n - mean*mean
        return mean, variance

    def is_overrepresented(self,cluster,clusterReps,frac,n):
        '''
            check whether cluster is overrepresented wrt. fract
            Args:
                cluster: selected cluster (int)
                clusterReps: list of selection likelihood of clusters 
                frac: maximal cluster representation fraction (float)
                n: number of instances to select (int)
            Returns:
                True or False
        '''
        if (clusterReps[cluster] <= frac):
            clusterReps[cluster] += 1.0/n
            return False
        else:
            return True
    
    def get_vector(self,default,length):
        '''
            get vector with length and default value
            Args:
                default: default value of each entry in vector
                length: length of vector
            Returns
                list /vector
        '''
        return length*[default]
    
    def print_samples(self,samples):
        '''
            print samples instances
            Args: 
                samples: list of instance names 
        '''
        print(">>>>>>>>>> Selected Instances ("+str(len(samples))+"):")
        for s in samples:
            print(s)
            
    def runtime_of_samples(self,samples):
        '''
            print runtimes of sampled instances and aggregation per solver (sum and #timeouts)
            Args:
                samples: list of instance names
        '''
        print("CSV of runtimes samples")
        sums = self.get_vector(0, len(self.runtimeData[0]))
        TOs = self.get_vector(0, len(self.runtimeData[0]))
        print(",".join(self.solvers)+",Min,Avg")
        for s in samples:
            times = self._runtime_data_dic[s]
            index = 0
            for t in times:
                if (t == self.cutoff):
                    TOs[index] += 1
                sums[index] += t
                index += 1  
            print(s+","+",".join(self.to_str_list(times))+","+str(min(times))+","+str(sum(times)/len(times)))
        print("SUM:"+","+",".join(self.to_str_list(sums)))
        print("Timeouts:"+","+",".join(self.to_str_list(TOs)))
        
    def features_of_samples(self,samples):
        '''
            print features of sampled instances
            Args:
                samples: list of instance names
        '''
        print("CSV of feature samples")
        for s in samples:
            feats = self.__feature_data_dic[s]
            print(s+","+",".join(self.to_str_list(feats)))
            
    def to_str_list(self,list):
        '''
            change list of type T to list of type str
            Args:
                list
            Returns:
                sList: string list
        '''
        sList = []
        for data in list:
            sList.append(str(data))
        return sList

    def print_stats(self):
        '''
            print some final stats of sampling
        '''
        print("Cluster Distribution:")
        print("")
        print("Complete Set")
        cluster_dist = {}
        for inst,cluster in self.__clusters.items():
            cluster_num = cluster_dist.get(cluster)
            if not cluster_num:
                cluster_num = 0
            cluster_num += 1
            cluster_dist[cluster] = cluster_num
        print(cluster_dist)
        print("")
        print("Selected Set")
        cluster_dist = {}
        for inst in self.__samples:
            cluster = self.__clusters[inst]
            cluster_num = cluster_dist.get(cluster)
            if not cluster_num:
                cluster_num = 0
            cluster_num += 1
            cluster_dist[cluster] = cluster_num
        print(cluster_dist)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog="selector.py",)
    req_group = parser.add_argument_group("Required Options")
    req_group.add_argument('--runtimes', dest='times', action='store', required=True, help='runtimes in csv (first col with instance names')    
    req_group.add_argument('--features', dest='feats', action='store', required=True, help='instance features in csv (first col with instance names')
    req_group.add_argument('--cutoff', dest='cutoff', action='store', type=int, required=True, help='cutoff time')
    req_group.add_argument('--n', dest='n', action='store', type=int, required=True, help='desired number of instances')

    opt_group = parser.add_argument_group("Optional Options")
  #  opt_group.add_argument('--cluster', dest='__clusters', action='store', type=int, default=10, help='number of desired __clusters in the feature space')    
    opt_group.add_argument('--reps', dest='reps', action='store', default=100, type=int, help='repetitions of kmeans clustering')   
    opt_group.add_argument('--frac', dest='frac', action='store', default=0.2, type=float, help='maximum representation of each cluster [0,1]')   
    opt_group.add_argument('--easyK', dest='easyK', action='store', default=0.1, type=float, help='remove too easy instances (solved by all solvers and avg runtimes < k*cutoff')   
    opt_group.add_argument('--aggregate', dest='agg', action='store', default="avg", choices=["avg","min","ind"], help='aggregation of instance runtimes')   
    opt_group.add_argument('--dist', dest='dist', action='store', default="gauss", choices=["gauss","uni","exp","log"], help='sample distribution')   
    opt_group.add_argument('--split', dest='split', action='store_true', default=False, help='before sampling, split randomly instance set in test and training set')   
    
    args = parser.parse_args()
    print(args)
    
    if(args.easyK*args.cutoff < 10.0):
        args.easy = 10.0/args.cutoff
    
    selector = Selector(args.cutoff)
    selector.parse_features(args.feats)
    selector.parse_runtimes(args.times)
    
    if args.split:
        selector.randomTestTrainingSplit()
    
    selector.joinTimesFeatures()
    selector.runtime_of_samples(selector._runtime_data_dic.keys())
    selector.removeTooEasy(args.cutoff, args.easyK)
    selector.clustering(args.reps)
    samples = selector.select(args.n, args.frac, args.agg, args.dist)
    selector.print_samples(samples)
    selector.runtime_of_samples(samples)
    selector.features_of_samples(samples)
    selector.print_stats()
    