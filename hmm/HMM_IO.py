#!/usr/bin/env python
'''
This file contains the input/output for the HMM.
The functions that read the human made HMM definitions and
their parsing.
In order to build an HMM from scratch we need this package
'''

# A copy of Fariselli's original code extended for Gaussian Mixture Models.
# Here the observations are not discrete symbols but rather continuous vectors of a signal
# or a sequence profile. The emission symbols (EMISSION_ALPHABET) are only used for the 
# human readable display. The dimension of the means and covariance matrices are determined 
# by including an integer value after the MIXTURES Yes tag, contained in the ".mod" grammar file header.
# 
# There are additional tags used to denote the GMM parameters. See SR_theorecticalModel.mod
# file for an example. 
# MIXTURE_NUM 3  --> denotes the number of mixtures in this hidden state
# MIXTURE_WEIGHTS w1 w2 w3 --> the coefs/weights for each mixture in this hidden state
# MIXTURE_MEANS mean1 mean2 mean3 --> the means for the mixtures in this hidden state 
#      where mean vectors are entered in order and the name for the text file containing 
#      the values is entered. Each mean will have shape (Ax1), where A is the number of 
#      components of the observation vectors and entered on one line as
#      m11 m12 m13 m14 ... m1A. The file will have filetype ".mean". 
#     
# MIXTURE_COVARS covar1 covar2 covar3 --> the covariance matrices for the mixtures 
#     in this hidden state, with a similar entry format as the means, see above. 
#     Each covariance matrix has the shape (AxA) and there will be MIXTURE_NUM of them. 
#     The indexing works by (row x col) with each mixture's covariance being entered in a file
#     covar111 ... covar11A ... covar1AA on multiple lines. Each line represents a row in the
#     covariance matrix. The file will have filetype ".cov".
#
# Provide a check counter for each of the parameters to determine incorrect input.


# Import NumPY for matrix/vector notation for mixture means and covariances
# It might be better to import this as a local library in the function used, rather
# than as a global library. 

import numpy

def save_cPickle(hmm,file):
    ''' save the hmm into cPickle file '''
    import cPickle
    fs=open(file,'w')
    cPickle.dump(hmm,fs)
    fs.close()

def load_cPickle(file):
    ''' load the object from file '''
    import cPickle
    import copy
    fs=open(file,'r')
    hmm=cPickle.load(fs)
    fs.close()
    return hmm

def parse_weight(name):
    '''
        Parse the textfile for a mixture's weights
    '''
    fweight = open(name).readlines()
    weights = []
    
    for line in fweight:
        line.rstrip()
        weights.append(float(line))
    
    return weights

def parse_mean(name,dimA):
    '''
        Parse the textfile for a mixture's mean vector, reads the vector down one column on multiple rows
    '''
    fmean = open(name).readlines()
    mean_vec = []
    
    for line in fmean:
        line.rstrip()
        mean_vec.append(float(line))
    #elems = fmean.readline().rstrip().split()
    #mean_vec = [float(e) for e in elems]
    return mean_vec

def parse_covar(name,dimA):
    '''
        Parse the textfile for a mixture's covariance matrix.
    '''
    covarf = open(name)    
    covar = numpy.zeros((dimA,dimA))
    i = 0    
    
    for row in covarf:
        col = row.rstrip().split()        
        for j in range(len(col)):
            covar[i][j] = float(col[j])
        i += 1
    covarf.close()
    return covar

def parse_text(text,profileDim,gmm_param_dir):
    ''' parse_text(text) legge 
        il contenuto dell'hmm passato attraverso text
        che a sua volta e` derivato da un file di tipo mod
        
        The profileDim parameter is the dimension of the input encoding used. For a gmmhmm not using mixtures 
        the profileDim will take on the default parameter value of zero.
    '''
    ret={} # return dictionary
    import sys
    import re
    import string   # so the functions used from this library have been deprecated in newer releases
    curr_name=None # nome corrente
    for line in text:
        if (re.search('^#',line) == None):
            list=line.split()
            if (len(list)>0):
                if(list[0] == 'TRANSITION_ALPHABET' or list[0] == 'EMISSION_ALPHABET'):
                    ret[list[0]]=list[1:]
                elif(list[0] == 'MIXTURES'):  # check if we are dealing a GMMHMM
                    ret[list[0]] = list[1:]
                    ret[list[0]][1] = profileDim #set the input encoding dimension externally from the grammar .mod file 
                elif(list[0] == 'NAME'):
                    curr_name=list[1]   # set the name of the current hidden state
                    ret[curr_name]={}
                    ret[curr_name].update({'FIX_TR':None})
                    ret[curr_name].update({'FIX_EM':None})
                    ret[curr_name].update({'FIX_EM_MIX':None})
                    if(curr_name not in ret['TRANSITION_ALPHABET']):   # check to ensure states are consistent
                        sys.exit(curr_name + " not in TRANSITION_ALPHABET ="+str(ret['TRANSITION_ALPHABET']))
                elif(list[0] == 'FIX_TR'):
                    ret[curr_name].update({'FIX_TR':'YES'})
                elif(list[0] == 'FIX_EM'):
                    ret[curr_name].update({'FIX_EM':'YES'})
                elif(list[0] == 'FIX_EM_MIX'):
                    ret[curr_name].update({'FIX_EM_MIX':'YES'})
                elif(list[0] == 'ENDSTATE'):
                    ret[curr_name].update({list[0]:int(list[1])})
                elif(list[0] == 'LABEL'):
                    ret[curr_name].update({list[0]:list[1]})
                elif(list[0] == 'TRANS'):
                    if(list[1] == 'None'):
                        tmplist=[]
                    else:
                        tmplist=list[1:]
                        for i in range(len(tmplist)):
                            try:
                                tmplist[i]=string.atof(tmplist[i])
                            except:
                                pass
                    ret[curr_name].update({list[0]:tmplist})
                elif(list[0] == 'LINK'):
                    if(list[1] == 'None'):
                        tmplist=[]
                    else:
                        tmplist=list[1:]
                    ret[curr_name].update({list[0]:tmplist})
                elif(list[0] == 'EMISSION'): # and ret['MIXTURES'][0] != 'Yes'): # Fariselli checks if EMISSION in a state is empty to see if it is a null/silent state
                    if(list[1] == 'None'):   # For GMM I will check if MIX_NUM == 0 for null/silent state
                        tmplist=[]
                    else:
                        tmplist=list[1:]
                        for i in range(len(tmplist)):
                            try:
                                tmplist[i]=string.atof(tmplist[i])
                            except:
                                pass
                    ret[curr_name].update({list[0]:tmplist})
                elif(list[0]=='EM_LIST'): # and ret['MIXTURES'][0] != 'Yes'): # emission list is left to denote the AA symbols
                    if(list[1] == 'None'):
                        tmplist=[]
                    else:
                        tmplist=list[1:]
                    ret[curr_name].update({list[0]:tmplist})
                elif(list[0] == 'MIXTURE_NUM'): # and ret['MIXTURES'][0] == 'Yes'): # number of mixtures in this state
                    if(list[1] == 'None'):
                        tmplist = []
                    else:
                        tmplist = list[1:]
                        for i in range(len(tmplist)):
                            try:
                                tmplist[i] = string.atoi(tmplist[i])
                            except:
                                pass
                    ret[curr_name].update({list[0]:tmplist})
                elif(list[0] == 'MIXTURE_WEIGHTS'): # and ret['MIXTURES'][0] == 'Yes'): # get the mixture weights
                    tmplist = []
                    if(list[1] == 'None'):
                        tmplist=[]
                    else:
                        weights = list[1:]
                        tmp_inner = parse_weight(gmm_param_dir+str(weights[0])+".weight")
                        tmplist = [float(w) for w in tmp_inner]
                        if(ret[curr_name]['MIXTURE_NUM'] != []) and (ret[curr_name]['MIXTURE_NUM'][0] != "tied") and (len(tmplist) != ret[curr_name]['MIXTURE_NUM'][0]): # check to ensure mixture weights are consistent
                            sys.exit(curr_name + " has incorrect number of mixture weights = " + str(ret[curr_name]['MIXTURE_NUM']))
                    ret[curr_name].update({list[0]:tmplist})
                elif(list[0] == 'MIXTURE_MEANS'): # and ret['MIXTURES'][0] == 'Yes'): # get the mixture means these are vectors of shape (Ax1) stored as a list of vectors
                    if (ret[curr_name]['MIXTURE_NUM'] != []) and (ret[curr_name]['MIXTURE_NUM'][0] != 'tied'):
                        dimA = int(ret['MIXTURES'][1])
                        tmplist = []
                        if(list[1] == 'None'):
                            tmplist=[]
                        else:
                            means = list[1:]
                            for i in range(ret[curr_name]['MIXTURE_NUM'][0]):
                                tmp_inner = parse_mean(gmm_param_dir+str(means[i])+".mean",dimA)
                                tmplist.append(tmp_inner) #tmplist[i] = [float(elem) for elem in list[i*dimA:(i+1)*dimA]]	 # a list of mean vectors
                                numelems = sum([len(mean_vec) for mean_vec in tmplist])
                            if (ret[curr_name]['MIXTURE_NUM'] != []) and (numelems % dimA != 0): # check to ensure mean components are consistent
                                sys.exit(curr_name + " has incorrect number of mixture mean components")
                    else:
                        tmplist = []
                    ret[curr_name].update({list[0]:tmplist})
                elif(list[0] == 'MIXTURE_COVARS'): # and ret['MIXTURES'][0] == 'Yes'): # get the mixture covariances, these are matrices with shape (AxA) stored as a list of matrices
                    if (ret[curr_name]['MIXTURE_NUM'] != []) and (ret[curr_name]['MIXTURE_NUM'][0] != 'tied'):
                        dimA = int(ret['MIXTURES'][1])
                        covars = list[1:]
                        tmplist = []
                        if(list[1] == 'None'):
                            tmplist=[]
                        else:
                            if (ret[curr_name]['MIXTURE_NUM'] != []) and (ret[curr_name]['MIXTURE_NUM'][0] != 'tied') and (len(covars) != ret[curr_name]['MIXTURE_NUM'][0]): # check to ensure covariance components are consistent in number
                                print "len(covar),dimA",len(covars),dimA
                                sys.exit(curr_name + " has incorrect number of mixture covariance components")
                            for covar_id in range(ret[curr_name]['MIXTURE_NUM'][0]):
                                tmp_covar = parse_covar(gmm_param_dir+str(covars[covar_id])+".covar",dimA)
                                tmplist.append(tmp_covar)
                    else:
                        tmplist = []
                    ret[curr_name].update({list[0]:tmplist})  # storing a list of numpy matrices
    return(ret)      

def write_for_humans(hmm,filename):
    ''' write_for_humans(hmm,filename) write on filename the hmm in the same format it is read by parse_text'''
    import sys	
    try:
        f=open(filename,'w')
    except:
        print "Can't open write_for_humans file, ", filename
        sys.exit()
    
    os_path = filename[:-10] + "/"
    
    strPrint=""
    separator="#############################\n"
    strPrint+="# alphabets\n"
    strPrint+="TRANSITION_ALPHABET "
    for i in hmm.topo_order:
        strPrint+=str(hmm.state_names[i])+" "
    strPrint+="\n"
    strPrint+="EMISSION_ALPHABET "+" ".join(hmm.emission_alphabet)+'\n'
    strPrint+="MIXTURES " + hmm.mixtures + " " + str(hmm.dimProfile) +'\n'
    strPrint+=separator
    for i in hmm.topo_order: # for each state in topological order
        strPrint+="NAME "+hmm.state_names[i]+'\n'
        strPrint+="LINK " #trransitions
        if(hmm.out_s[i]):
            for j in hmm.out_s[i]:
                strPrint+=hmm.state_names[j]+' '
            strPrint+='\n'
        else:
            strPrint+=" None\n"
        
        strPrint+="TRANS "
        if(hmm.states[i].tied_t):
            strPrint+="tied "+hmm.states[i].tied_t+'\n'
        elif(hmm.out_s[i]):
            for j in hmm.out_s[i]:
                strPrint+=str(hmm.states[i].a(hmm.states[j]))+' '
            strPrint+='\n'
        else:
            strPrint+=" None\n"
        if(hmm.fix_tr[i]):
            strPrint+="FIX_TR\n"
        # end state
        
        strPrint+="ENDSTATE "+str(hmm.states[i].end_state)+'\n'
        #emissions
        if(not hmm.states[i].is_null()):   #if (hmm.states[i].em_letters):
            strPrint+="EM_LIST "+" ".join(hmm.states[i].em_letters)+'\n'
        else:
            strPrint+="EM_LIST None\n"
        
        strPrint+="EMISSION "
        if(hmm.states[i].tied_e): # and hmm.mixtures != "Yes"):
            strPrint+="tied "+hmm.states[i].tied_e+'\n'
        elif(not hmm.states[i].is_null() and hmm.mixtures != "Yes"): #elif(hmm.states[i].em_letters):
            v=hmm.states[i].get_emissions()
            for j in v:
                strPrint+=str(j)+' '
            strPrint+='\n'
        else:
            strPrint+=" None\n"
        
        #gaussian mixture model parameters
        strPrint+= "MIXTURE_NUM " 
        if(not hmm.states[i].is_null() and hmm.mixtures == "Yes"):
            if hmm.states[i].tied_e_mix and not hmm.states[i].tied_e:
                strPrint+="tied "+hmm.states[i].tied_e_mix+'\n'
            elif not hmm.states[i].tied_e_mix and not hmm.states[i].tied_e:
                strPrint += str(hmm.states[i].get_emissions().get_mix_num()) + '\n'
            else:
                strPrint += "None\n"
        else:
            strPrint += "None\n"
        
        strPrint+= "MIXTURE_WEIGHTS " 
        if(not hmm.states[i].is_null() and hmm.mixtures == "Yes"):
            if not hmm.states[i].tied_e:
                weights = []
                for mix in range(hmm.states[i].get_emissions().get_mix_num()):
                    weights.append(hmm.states[i].get_emissions().get_mix_weight(mix))
                #    weight_name = str(hmm.states[i].name) + str(hmm.states[i].get_em_name()[3:]) + "_mix_%d"%mix + ".weight" 
                #    strPrint += weight_name[:-7] + ' '
                strPrint += str(hmm.states[i].name) + str(hmm.states[i].get_em_name()[3:])
                write_weight(os_path + str(hmm.states[i].name) + str(hmm.states[i].get_em_name()[3:]) + ".weight",weights)
                strPrint +="\n"
            else:
                strPrint += "None\n"
        else:
            strPrint += "None\n"
        
        strPrint+= "MIXTURE_MEANS "
        if(not hmm.states[i].is_null() and hmm.mixtures == "Yes"):
            if not hmm.states[i].tied_e and not hmm.states[i].tied_e_mix:
                mix_num = hmm.states[i].get_emissions().get_mix_num()
                for mix in range(mix_num):
                    mean = hmm.states[i].get_emissions().get_mixture_density(mix).get_mean()
                    mean_name = str(hmm.states[i].name) + str(hmm.states[i].get_em_name()[3:]) + "_mix_%d"%mix + ".mean"                 
                    strPrint += mean_name[:-5] + " "                
                    write_mean(os_path + mean_name,mean)
                strPrint += "\n"
            else:
                strPrint += "None\n"
        else:
            strPrint += "None\n"
        
        strPrint+= "MIXTURE_COVARS " 
        if(not hmm.states[i].is_null() and hmm.mixtures == "Yes"):
            if not hmm.states[i].tied_e and not hmm.states[i].tied_e_mix: 
                mix_num = hmm.states[i].get_emissions().get_mix_num()
                for mix in range(mix_num):
                    covar = hmm.states[i].get_emissions().get_mixture_density(mix).get_cov() #this is a numpy array shape (dimProfile,dimProfile)
                    covar_name = str(hmm.states[i].name) + str(hmm.states[i].get_em_name()[3:]) + "_mix_%d"%mix + ".covar"                 
                    strPrint += covar_name[:-6] + " "                
                    write_covar(os_path + covar_name,covar,hmm.dimProfile)                
                strPrint +="\n"
            else:
                strPrint += "None\n"
        else:
            strPrint += "None\n"
        
        if(hmm.fix_em[i]):
            strPrint+="FIX_EM\n"
        
        if(hmm.fix_em_mix[i]):
            strPrint+="FIX_EM_MIX\n"
        
        #labels
        if(hmm.states[i].label):
            strPrint+="LABEL "+hmm.states[i].label+'\n'
        else:
            strPrint+="LABEL None\n"
        
        strPrint+=separator
    
    f.write(strPrint)
    f.close()        

def write_weight(filename,weights):
    '''
        Writes a mixture's weight to a file output.
    '''
    fweight = open(filename,'w')                
    weightstr = ""
    for w in weights:
        weightstr += str(w) + "\n"
    fweight.write(weightstr)
    fweight.close()

def write_mean(filename,mean):
    '''
        Writes a mixture's mean vector to a file output.
    '''
    fmean = open(filename,'w')                
    meanstr = ""                
    for comp in mean:
        meanstr += str(comp) + "\n"
    fmean.write(meanstr)
    fmean.close()
    
def write_covar(filename,covar,dimA):
    '''
        Writes a mixture's covariance matrix to a file output.
    '''
    fcovar = open(filename,'w')                
    covarstr = ""                
    for i in range(len(covar.flat)):
        covarstr += str(covar.flat[i]) + " "
        if (i+1)%dimA == 0:
            covarstr += "\n"
    fcovar.write(covarstr)
    fcovar.close()
             
def Build_HMM(file,profileDim,gmm_param_dir):
    ''' 
        This function build an hmm starting from a file. It discriminates between a discrete symbol
        version and a GMM version using different node_em class objects. Also searches for 
        MIXTURES = Yes/No indicator from the input file.
        The profileDim parameter is the dimension of the input encoding used.
    '''
    import string
    import State
    import HMM
    
    try:
        lines = open(file).readlines()
    except:
        print "Can't open build_hmm file, ",file
    
    info=parse_text(lines,profileDim,gmm_param_dir)
    tr_al=info['TRANSITION_ALPHABET']
    
    if(info['EMISSION_ALPHABET'][0] == 'range'): # used for integer emission symbols such as dice throws
        em_al=[]
        for i in range(string.atoi(info['EMISSION_ALPHABET'][1])):
            em_al.append(str(i))
    else:
        em_al=info['EMISSION_ALPHABET']
    
    tied_t={} # tied transitions None if not tied
    tied_e={} # tied emissions None if not tied (this is at the state level)
    tied_e_mix = {} # tied emission mixture densities, None if not tied (this is at the sub-state mixture-tying level)
    links={} # temporary list for each state 
    states=[]
    label={}
    endstate={}
    in_links={}
    fix_tr={}
    fix_em={}
    fix_em_mix = {}
    empty_tr=State.node_tr("_EMPTY_TR_",[])  #empty transition and emissions
    
    if info["MIXTURES"][0] == "Yes":  # check if node_em is for discrete symbols or vector profile GMMs
        empty_em = State.node_em_gmm("_EMPTY_EM_",[],[]) # mix_weights = [], mix_densities = []
    else:
        empty_em = State.node_em("_EMPTY_EM_",[])
    
    for name in tr_al: # initialize dictionaries
        in_links[name]=[]
        links[name]=[None,None]
    
    for name in tr_al: # create in_link information
        for in_name in info[name]['LINK']:
            if(name not in in_links[in_name]):
                in_links[in_name].append(name)
    
    serial=0 # used as incremental internal number for transitions and emissions. It will be used toi set node_tr node_em
    for name in tr_al: # set node_tr
        if(info[name]['TRANS']!=[] and info[name]['TRANS'][0] != 'tied'):
            if(info[name]['TRANS'][0] == 'uniform'): # set uniform
                d=1.0/len(info[name]['LINK'])
                info[name]['TRANS']=[d]*len(info[name]['LINK'])
            obj=State.node_tr("_TR_"+str(serial),info[name]['TRANS'])
            serial=serial + 1
            links[name][0]=obj
            tied_t[name]=None
        elif(info[name]['TRANS']!=[] and info[name]['TRANS'][0] == 'tied'):
            tmpname=info[name]['TRANS'][1]
            links[name][0]=links[tmpname][0] # links[name][0] is for transitions
            tied_t[name]=tmpname
        elif(info[name]['TRANS']==[]):
            links[name][0]=empty_tr
            tied_t[name]=None
    
    # This section implements the tying of emission density functions (either discrete/continuous at the state level or for GMM at the sub-state mixture level)
    # For mixture-tying, the implementation is such that a collection of states will share one codebook of mixture densities while still having individual state 
    # mixture weights
    serial=0
    for name in tr_al: # set node_em
        if (info["MIXTURES"][0] != "Yes"):
            if(info[name]['EMISSION']!=[] and info[name]['EM_LIST'][0] == 'all'):
                info[name]['EM_LIST']=em_al
            elif(info[name]['EMISSION']!=[] and info[name]['EMISSION'][0] != 'tied'):
                if(info[name]['EMISSION'][0] == 'uniform'): # set uniform
                    d=1.0/len(info[name]['EM_LIST'])
                    info[name]['EMISSION']=[d]*len(info[name]['EM_LIST'])
                obj=State.node_em("_EM_"+str(serial),info[name]['EMISSION'])
                serial=serial + 1
                links[name][1]=obj
                tied_e[name]=None
                tied_e_mix[name] = None
            elif(info[name]['EMISSION']==[]):
                links[name][1]=empty_em
                tied_e[name]=None
                tied_e_mix[name] = None
            elif(info[name]['EMISSION']!=[] and info[name]['EMISSION'][0] == 'tied'):
                # check for states with tied emissions discrete and GMM at the state level
                tmpname=info[name]['EMISSION'][1]
                links[name][1]=links[tmpname][1]
                tied_e[name]=tmpname
                tied_e_mix[name] = tmpname
        elif(info["MIXTURES"][0] == "Yes"):
            if(info[name]['EMISSION']!=[] and info[name]['EMISSION'][0] == 'tied'):
                # check for states with tied emissions discrete and GMM at the state level
                tmpname=info[name]['EMISSION'][1]
                links[name][1]=links[tmpname][1]
                tied_e[name]=tmpname
                tied_e_mix[name] = tmpname
            elif (info[name]['MIXTURE_NUM'] != []) and (info[name]['MIXTURE_NUM'][0] != 'tied'):
                #normalise mixture weights to sum to one
                weight_sum = float(sum(info[name]["MIXTURE_WEIGHTS"]))
                info[name]["MIXTURE_WEIGHTS"] = [w/weight_sum for w in info[name]["MIXTURE_WEIGHTS"]]
                mix_densities = []
                for k in range(int(info[name]["MIXTURE_NUM"][0])):
                    #tmp_mix_density = State.mixture_density("_EM_GMM_"+str(serial)+"_MIX_"+str(k),info[name]["MIXTURE_MEANS"][k],info[name]["MIXTURE_COVARS"][k])
                    #all mixture densities in a state share the same name, this is because tying of mixture densities works by tying all mixtures in a state to another state.
                    tmp_mix_density = State.mixture_density("_EM_GMM_"+str(serial)+"_MIX",info[name]["MIXTURE_MEANS"][k],info[name]["MIXTURE_COVARS"][k])
                    mix_densities.append(tmp_mix_density)
                obj = State.node_em_gmm("_EM_GMM_"+str(serial),info[name]["MIXTURE_WEIGHTS"],mix_densities)
                serial=serial+1
                links[name][1]=obj
                tied_e[name]=None
                tied_e_mix[name] = None
            elif (info[name]['MIXTURE_NUM'] == []):
                links[name][1] = empty_em
                tied_e[name] = None
                tied_e_mix[name] = None
            elif (info[name]['MIXTURE_NUM'][0] == 'tied'):
                # check for states with tied emissions GMM at the sub-state mixture level
                #normalise mixture weights to sum to one
                weight_sum = float(sum(info[name]["MIXTURE_WEIGHTS"]))
                info[name]["MIXTURE_WEIGHTS"] = [w/weight_sum for w in info[name]["MIXTURE_WEIGHTS"]]
                tmpname = info[name]["MIXTURE_NUM"][1]
                #this is used to obtain the reference pointer to the tied to state's mixture densities
                tmp_node_em_gmm = links[tmpname][1]
                tied_mixture_densities = tmp_node_em_gmm.get_mixtures()
                obj = State.node_em_gmm("_EM_GMM_"+str(serial),info[name]["MIXTURE_WEIGHTS"],tied_mixture_densities)
                serial=serial+1
                links[name][1] = obj
                tied_e[name]= None
                tied_e_mix[name]=tmpname
    
    for name in tr_al: # set labels 
        if(info[name]['FIX_TR']): # fixed transitions
            fix_tr[name]='YES'
        else:
            fix_tr[name]=None
        if(info[name]['FIX_EM']): # fixed emissions
            fix_em[name]='YES'
        else:
            fix_em[name]=None
        if(info[name]['FIX_EM_MIX']): # fixed emission mixture densities
            fix_em_mix[name]='YES'
        else:
            fix_em_mix[name]=None
        
        if(info[name]['LABEL'] == ['None']): # LABELS
            label[name]=None
        else:
            label[name]=info[name]['LABEL']
        endstate[name]=info[name]['ENDSTATE'] # set endstates
        states.append(State.State(name,links[name][0],links[name][1],info[name]['LINK'],in_links[name],info[name]['EM_LIST'],tied_t[name],tied_e[name],tied_e_mix[name],endstate[name],label[name])) # set State[i] and appnd it to the state list
    
    hmm = HMM.HMM(states,em_al,fix_tr,fix_em,fix_em_mix,info["MIXTURES"][0],profileDim) #int(info["MIXTURES"][1])) # set self.hmm => the HMM
    return(hmm)

def main():
    print "Test with file mod.mod"
    print "read file and create an hmm"
    hmm = Build_HMM("Test_Case2.mod") #need to include gmm_parameter directory and input encoding dimension
    print "inlinks (<-)  - outlinks (->) "
    for i in range(hmm.num_states):
        print hmm.states[i].name," -> ",hmm.states[i].out_links, " <- ",hmm.states[i].in_links
    print "print the tied states"
    for i in range(hmm.num_states):
        n=hmm.states[i]
        if(n.tied_e):
            print "State ",n.name," has tied emission to ",n.tied_e
        elif (m.tied_e_mix):
            print "State ",n.name," has tied emission mixture densities to ",n.tied_e_mix
        if(n.tied_t):
            print "State ",n.name," has tied transitions to ",n.tied_t
    print "###############################################################"
    print"# state    tr_name  em_name  # state     tr_name  em_name   tying test       tying test"   
    for i in range(hmm.num_states):
        ni=hmm.states[i]
        for j in range(i,hmm.num_states):
            nj=hmm.states[j]
            print "# <",ni.name,"> ",ni._node_tr.name,ni._node_em.name,
            print "# <",nj.name,"> ",nj._node_tr.name,nj._node_em.name,
            print " (TR1==TR2)? ",(hmm.states[i]._node_tr == hmm.states[j]._node_tr),
            print " (EM1==EM2)? ",(hmm.states[i]._node_em == hmm.states[j]._node_em)
    print "\n===================================Topo_order============================="
    for i in hmm.topo_order:
        print i,hmm.state_names[i]
    # test case for the input and output of the hmm grammar file parsing
    write_for_humans(hmm,"Test_Case_Output.mod")

if __name__ == '__main__':
	main()