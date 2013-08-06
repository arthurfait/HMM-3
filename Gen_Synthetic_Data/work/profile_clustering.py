#!/usr/bin/env python



import sys, os, cmd
import numpy as NUM
from sklearn import mixture

sys.path.append('../')

import GMMHMM.nearPD_Higham

###############################################################
###############################################################

n_mix = 1 # This must match up to the number of mixtures defined in the grammar .mod file

if len(sys.argv)<3:
   print "usage :",sys.argv[0],"dataset, profileset, GMM_param output dir"
   sys.exit(-1)

pset=open(sys.argv[1]).readlines()

# read all proteins
prot_dict = {}
for protname in pset:
    prot=open(sys.argv[2]+protname.strip()+".profile",'r').readlines()
    #transform current protein sequence into window based representation
    seq=[]
    true_path = []
    for line in prot:
        v=line.split()
        listain=[float(x) for x in v[:-1]]
        true_path.append(str(v[-1].lower()))  #last element is the path label
        seq.append(listain)
    
    #partition each residue profile window into one of the TM_annotation labels
    for i in range(len(true_path)):
        label = true_path[i]
        curr_arr = prot_dict.setdefault(label,None)
        if curr_arr == None:
            #first element
            new_elem = NUM.array([float(x) for x in seq[i]])
            curr_arr = NUM.array([new_elem])    
        else:
            #previous elements
            new_elem = NUM.array([float(x) for x in seq[i]])
            curr_arr = NUM.concatenate((curr_arr,[new_elem]))
        prot_dict[label] = curr_arr


for key,data in prot_dict.items():
    #print "TM_Annotation,profile vectors = ",key,data
    #clf = mixture.GMM(n_components=n_mix, covariance_type='diag') 
    clf = mixture.GMM(n_components=n_mix, covariance_type='full') 
    #clf = mixture.VBGMM(n_components=n_mix, covariance_type='full') 
    # This can lead to runtime errors if there is only one sample and there are more mixtures
    clf.fit(data)
    #print NUM.round(clf.weights_, 2)
    #print NUM.round(clf.means_, 2)
    #print NUM.round(clf.covars_, 2) 
    
    NUM.savetxt(sys.argv[3]+'seq_profile_GMM_data_weights_'+key+".weight", clf.weights_, delimiter=' ', fmt='%1.4e')
    for m in range(n_mix):
        dimA = len(clf.means_[m])
        #tmp_covar = NUM.identity(dimA)
        tmp_covar = NUM.zeros((dimA,dimA))
        #below used for diagonal initialised covar matrices
        #for i in range(dimA):
        #    tmp_covar[i,i] = clf.covars_[m][i]
        #below used for full initialised covar matrices
        for i in range(dimA):
            for j in range(dimA):
                tmp_covar[i,j] = clf.covars_[m][i][j] # used for normal GMM clustering
                #tmp_covar[i,j] = NUM.linalg.pinv(clf.precs_[m][i][j]) # used for VBGMM clustering
        if NUM.linalg.det(tmp_covar) == 0.0:
            #find nearest matrix which is positive definite
            pd_tmp = GMMHMM.nearPD_Higham.nearPD(tmp_covar)
            tmp_covar = pd_tmp
            print "used PD_Higham"
        NUM.savetxt(sys.argv[3]+"seq_profile_GMM_data_means_"+str(m+1)+"_"+key+".mean", clf.means_[m], delimiter=' ',fmt='%1.4e')
        NUM.savetxt(sys.argv[3]+"seq_profile_GMM_data_covars_"+str(m+1)+"_"+key+".covar", tmp_covar, delimiter=' ',fmt='%1.4e')
        #print "det covar = ",NUM.linalg.det(tmp_covar)
        