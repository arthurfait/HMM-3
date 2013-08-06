#!/usr/bin/env python


import sys, os, cmd
import numpy as NUM
sys.path.append('../') #need to access GMMHMM directory and python libraries

import GMMHMM.tr_obj as tr_obj
import GMMHMM.HMM_IO as HMM_IO
import GMMHMM.algo_HMM as algo_HMM
import GMMHMM.HMM as HMM


if len(sys.argv)<7:
   print "usage :",sys.argv[0],"HMM_Grammar_file,HMM_params_dir,profile_dimension, max_seq_len, num_sequences, gen_seq output dir, set_name"
   sys.exit(-1)

hmm_grammar = sys.argv[1]
hmm_params_dir = sys.argv[2] # this directory should already exist with parameters
profileDim = int(sys.argv[3])
max_seq_len = int(sys.argv[4])
num_seqs = int(sys.argv[5])
gen_seq_outdir = sys.argv[6]
set_name = sys.argv[7]

# read in predefined grammar and parameters for the GMMHMM
print "Build_HMM"
hmm=HMM_IO.Build_HMM("../sets/"+str(hmm_grammar)+'.mod.gen',profileDim, str(hmm_params_dir))

# generate input sequence profiles
# use randomWalkGMM to generate some protein sequences

gen_set = open("../sets/"+str(set_name),'w')
gen_str = ""
for i in range(num_seqs):
    generated_seq = hmm.randomWalkGMM(max_seq_len)
    file=open(str(gen_seq_outdir)+"rand_"+str(hmm_grammar[2:])+"_"+str(i)+'.seq','w')
    file_profile=open(str(gen_seq_outdir)+"rand_"+str(hmm_grammar[2:])+"_"+str(i)+'.profile','w')
    output_str = ""
    output_str_profile = ""
    k = 0
    for (obs,label,state) in generated_seq:
        output_str += " ".join([str(e) for e in obs.flatten().tolist()]) + " " + str(label) + " " + str(state) + "\n"
        output_str_profile += " ".join([str(e) for e in obs.flatten().tolist()]) + " " + str(label) + "\n"
        k += 1
    file.write(output_str)
    file.close()
    file_profile.write(output_str_profile)
    file_profile.close()
    gen_str += "rand_"+str(hmm_grammar[2:])+"_"+str(i)+'\n'
    print "Rand_Seq_%s_%s has %s residues\n"%(str(hmm_grammar[2:]),i,k)

gen_set.write(gen_str)
gen_set.close()

print "finished generating sequences %s"%str(set_name)
