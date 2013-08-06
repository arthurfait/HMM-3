#!/usr/bin/env python

import sys, os, cmd
import numpy as NUM
sys.path.append('../') #need to access GMMHMM directory and python libraries


import GMMHMM.tr_obj as tr_obj
import GMMHMM.HMM_IO as HMM_IO
import GMMHMM.algo_HMM as algo_HMM



if len(sys.argv)<2:
   print "usage :",sys.argv[0],"dataset, grammar mod file"
   sys.exit(-1)

# read all proteins
trobjlist = []
prot=open(sys.argv[1] +".profile",'r').readlines()
protname = sys.argv[1].split("/")[-1]
#transform current protein sequence into window based representation
seq=[]
true_path = []
for line in prot:
    v=line.split()
    listain=[float(x) for x in v[:-1]]
    true_path.append(str(v[-1].lower()))  #last element is the path label
    seq.append(listain)    
trobjlist.append(tr_obj.TR_OBJ(seq,labels=true_path,name=protname.strip()))  

profileDim = len(trobjlist[0].seq[0]) # obtain the dimension of the input encoding
hmm=HMM_IO.Build_HMM(sys.argv[2]+'.mod.final',profileDim, sys.argv[2]+"/")
for o in trobjlist:
    (bestpath,bestval) = algo_HMM.ap_viterbi(hmm, o.seq, Scale='y')
    #(bestpath,bestval,logProbPath) = algo_HMM.viterbi(hmm, o.seq, returnLogProb=True, labels=None)
    fname = "../io/"+o.name+".apv.pred"
    file=open(fname,'w')
    pos=1
    file.write("# pos obs hmm_apv match\n")
    #fname = "../io/"+o.name+".vit.pred"
    #file=open(fname,'w')
    #pos=1
    #file.write("# pos obs hmm_vit match\n")
    for i in range(o.len): 
        #print hmm.states[bestpath[i]].name
        ind_tag = ""
        if bestpath[i] in hmm.emits: 
            if str(hmm.states[bestpath[i]].label) == o.labels[i]:
                ind_tag = '*'
            file.write(str(pos)+'\t'+o.labels[i]+'\t'+str(hmm.states[bestpath[i]].label)+'\t'+ind_tag+'\n')
            pos+=1
    file.close()
