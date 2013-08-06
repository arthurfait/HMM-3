#!/usr/bin/env python


import sys, os, cmd
import numpy as NUM
sys.path.append('../') #need to access GMMHMM directory and python libraries

import GMMHMM.tr_obj as tr_obj
import GMMHMM.HMM_IO as HMM_IO
import GMMHMM.algo_HMM as algo_HMM

if len(sys.argv)<3:
   print "usage :",sys.argv[0],"dataset, profileset, grammar output dir"
   sys.exit(-1)

pset=open(sys.argv[1]).readlines()

# read all proteins
trobjlist = []
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
    
    trobjlist.append(tr_obj.TR_OBJ(seq,labels=true_path,name=protname.strip()))  
  
#train gmmhmm
print "Build_HMM"
profileDim = len(trobjlist[0].seq[0]) # obtain the dimension of the input encoding
hmm=HMM_IO.Build_HMM(sys.argv[1]+'.mod',profileDim, sys.argv[3]+"/")
(pseudoAC,pseudoWC,pseudoMC,pseudoUC) = (0.0, 1.0, 0.5, 1.0)
print "Baum_Welch"
algo_HMM.Baum_Welch(hmm,trobjlist,pseudoAC,pseudoWC,pseudoMC,pseudoUC,Scale='Yes',labels=None,maxcycles=10,tolerance=1e-9,verbose='y')
print "HMM write for humans."
HMM_IO.write_for_humans(hmm, sys.argv[3]+'.mod.final')


print "finished training"
mis_train = 0
total = 0
#run gmmhmm on training set
for i in range(len(trobjlist)):
    o = trobjlist[i]
    #print "NAME ",o.name
    #print "seqobject ",o,o.seq
    (bestpath,bestval) = algo_HMM.ap_viterbi(hmm, o.seq, Scale='y')
    #(bestpath,bestval,logProbPath) = algo_HMM.viterbi(hmm, o.seq, returnLogProb=True, labels=None)
    file=open(sys.argv[3]+"/"+o.name+'.apv.train','w')
    pos=1
    file.write("# pos obs hmm_apv match\n")
    #file=open(sys.argv[3]+"/"+o.name+'.vit.train','w')
    #pos=1
    #file.write("# pos obs hmm_vit match\n")
    for i in range(o.len): 
        #print hmm.states[bestpath[i]].name
        ind_tag = ""
        if bestpath[i] in hmm.emits: 
            if str(hmm.states[bestpath[i]].label) == o.labels[i]:
                ind_tag = '*'
            else:
                mis_train += 1
            total += 1
            file.write(str(pos)+'\t'+o.labels[i]+'\t'+str(hmm.states[bestpath[i]].label)+'\t'+ind_tag+'\n')
            pos+=1
    file.close()
#print "Mis-classification for training (%d out of %d)"%(mis_train,total)

