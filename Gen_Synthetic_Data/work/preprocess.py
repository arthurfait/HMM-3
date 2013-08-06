#!/usr/bin/env python


import sys, os, cmd
import numpy as NUM
sys.path.append('../') #need to access GMMHMM directory and python libraries

from sklearn.decomposition import PCA, KernelPCA
from sklearn.decomposition import ProjectedGradientNMF
import GMMHMM.tr_obj as tr_obj
import GMMHMM.HMM_IO as HMM_IO
import GMMHMM.algo_HMM as algo_HMM

if len(sys.argv)<5:
   print "usage :",sys.argv[0]," window size, train_dataset, test_dataset, profileset, pca_dataset output dir"
   sys.exit(-1)

pset=open(sys.argv[2]).readlines()
win = int(sys.argv[1])
halwin=int(win/2)

# read all proteins from training set
trobjlist = []
pre_pca_lst = []
for protname in pset:
    prot=open(sys.argv[4]+protname.strip()+".profile",'r').readlines()
    seq=[]
    true_path = []
    seqtmp=[]
    for line in prot:
        v=line.split()
        if (v[-1].lower() == 'f') or (v[-1].lower() == 'b'):
            listain=[float(x) for x in v[:-1]]
            slist=sum(listain)
            if slist >0:
                listain=[x/slist for x in listain]
            else:
                listain=[1.0/len(listain) for x in listain]
            seqtmp.append(listain)
            true_path.append(str(v[-1].lower()))  #last element is the path label
        #else:
        #    print "Training Prot: %s, label: %s"%(protname.strip(),v[-1].lower())
    unif=[1.0/len(listain) for x in listain] #add uniformly distributed freq for extra half windows
    for i in range(halwin): #insert the extra half windows at the termini
        seqtmp.insert(0,unif)
        seqtmp.append(unif)
    for i in range(halwin,len(seqtmp)-halwin):
        v=[]
        for j in range(-halwin,halwin+1):
            v+=seqtmp[i+j]
        seq.append(v)
        pre_pca_lst.append(v)
    trobjlist.append(tr_obj.TR_OBJ(seq,labels=true_path,name=protname.strip()))  

# transform all training set proteins using PCA, ensure a maximum number of sample feature is 200
#pca_trans = PCA(n_components='mle')
pca_trans = KernelPCA(n_components=200,kernel="rbf", gamma=10)
pca_lst = pca_trans.fit_transform(pre_pca_lst)
#if len(pca_trans.explained_variance_ratio_) > 200:
#    pca_trans = PCA(n_components=200)
#    pca_lst = pca_trans.fit_transform(pre_pca_lst)

#print "Variance ratio explained by each PCA component",pca_trans.explained_variance_ratio_
#print "Number of PCA components %s"%len(pca_trans.explained_variance_ratio_)

'''
# non-negative matrix factorization using projected gradient
nmf_trans = ProjectedGradientNMF(n_components=200)
nmf_lst = nmf_trans.fit_transform(pre_pca_lst)
pca_trans = nmf_trans
pca_lst = nmf_lst
# Frobenius norm of the matrix difference between the training data and the reconstructed 
# data from the fit produced by the model. || X - WH ||_2
print "Reconstruction Error ",nmf_trans.reconstruction_err_
'''
# write newly PCA transformed input encodings to file, including original annotations
# the elements of pca_lst should be in the same positional ordering as trobjlist
cur_start = 0 #start index position for current protein sequence
for i in range(len(trobjlist)):
    o = trobjlist[i]
    cur_len = o.len
    cur_seq_str = ""
    cur_pca = pca_lst[cur_start:(cur_start+o.len)]
    for j in range(o.len):
        cur_seq_str += " ".join([str(e) for e in cur_pca[j]]) + " " + o.labels[j] + "\n"
    fout = open(sys.argv[5] + o.name + ".profile",'w')
    fout.write(cur_seq_str)
    fout.close()
    cur_start += o.len

# use the current PCA transformation to transfor the test data
pset=open(sys.argv[3]).readlines()
win = int(sys.argv[1])
halwin=int(win/2)

# read all proteins from training set
testobjlist = []
pre_pca_lst = []
for protname in pset:
    prot=open(sys.argv[4]+protname.strip()+".profile",'r').readlines()
    seq=[]
    true_path = []
    seqtmp=[]
    for line in prot:
        v=line.split()
        if (v[-1].lower() == 'f') or (v[-1].lower() == 'b'):
            listain=[float(x) for x in v[:-1]]
            slist=sum(listain)
            if slist >0:
                listain=[x/slist for x in listain]
            else:
                listain=[1.0/len(listain) for x in listain]
            seqtmp.append(listain)
            true_path.append(str(v[-1].lower()))  #last element is the path label
        #else:
        #    print "Testing Prot: %s, label: %s"%(protname.strip(),v[-1].lower())
    unif=[1.0/len(listain) for x in listain] #add uniformly distributed freq for extra half windows
    for i in range(halwin): #insert the extra half windows at the termini
        seqtmp.insert(0,unif)
        seqtmp.append(unif)
    for i in range(halwin,len(seqtmp)-halwin):
        v=[]
        for j in range(-halwin,halwin+1):
            v+=seqtmp[i+j]
        seq.append(v)
        pre_pca_lst.append(v)
    testobjlist.append(tr_obj.TR_OBJ(seq,labels=true_path,name=protname.strip()))

# transform all testing set proteins using PCA
pca_lst = pca_trans.transform(pre_pca_lst)

# write newly PCA transformed input encodings to file, including original annotations
# the elements of pca_lst should be in the same positional ordering as trobjlist
cur_start = 0 #start index position for current protein sequence
for i in range(len(testobjlist)):
    o = testobjlist[i]
    cur_len = o.len
    cur_seq_str = ""
    cur_pca = pca_lst[cur_start:(cur_start+o.len)]
    for j in range(o.len):
        cur_seq_str += " ".join([str(e) for e in cur_pca[j]]) + " " +o.labels[j] + "\n"
    fout = open(sys.argv[5] + o.name + ".profile",'w')
    fout.write(cur_seq_str)
    fout.close()
    cur_start += o.len