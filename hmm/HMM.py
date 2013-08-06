'''
This file contains the HMM definition
'''
try:
   import psyco
   psyco.full()
except:
   pass

import numpy as NUM
import State
import sys
from Def import DEF

ARRAYFLOAT=NUM.float64
ARRAYINT=NUM.int

def _sample_gaussian(mean, covar, n_samples=1):
    '''
        Generate random samples from a Gaussian distribution.
    '''
    n_dim = len(mean)
    rand = NUM.random.randn(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)
    
    from scipy import linalg
    U, s, V = linalg.svd(covar)
    sqrtS = NUM.diag(NUM.sqrt(s))
    sqrt_covar = NUM.dot(U, NUM.dot(sqrtS, V))
    rand = NUM.dot(sqrt_covar, rand)

    return (rand.T + mean).T
    
    
def _getRand(randint,values,names,norm=10000):
    ''' _getRand(randint,values,names,norm=10000) '''
    	#If you want to use the randomWalk function below, this _getRand() function
    	#must be modified to work with GMMs.
    if type(values) is list:
        #transition list or discrete emission list
        v=[]
        s=0
        for e in values: 
            s=int(norm*e+1)+s
            v.append(s)
        irand=randint(0,v[-1]-1)
        i=0
        while v[i] < irand:
            i=i+1
        return i,names[i]
    else:
         #continuous emissions with gaussian mixture model object. need to sample from object.
         print "continuous emission gaussian mixtures"

##########################################
class HMM:
    ''' define a HMM
        a HMM object consists of:
                                      
		self.num_states 		# number of states
        self.states	 		# the states
        self.emission_alphabet
        self.state_names		# the state names
        self.fix_tr 	# the transitions which are not be trained
         
        self.fix_em	# the emissions which are not trained
        --> these are fixed per state, ie all emission probs in a state are fixed!!!
        self.fix_em_mix # only the mixture densities in a state are fixed
        
        self.topo_order # the topologicrom al ordered state list (in numbers from 0)
        self.emits	# the list of emitting states
        self.nulls	# the list of the null states
        # list of the inlinks and outlinks for each states
        # also restricted to nulls and emitting
        self.out_s	# the outlinks of each state
        self.out_s_e	# the outlinks restricted to emitting states for each state
        self.out_s_n	# the outlinks restricted to null states for each state 
        self.in_s	# the inlinks of each state
        self.in_s_e	# the inlinks restricted to emitting states for each state 
        self.in_s_n	# the inlinks restricted to null states for each state 
        # end states, all, nulls and emitting
        self.end_s	# the list of the states that can end
        self.end_s_e	# the above list restricted to the emittings
        self.end_s_n	# the above list restricted to nulls
        # now we eliminate 0 from null states Begin must be a null state!!
        self.nulls	# the list of the null states without Begin
        self.label_list # the list of the labels of all states 
        self.effective_tr # the list of the effective transitions (No node_tr)
        self.effective_em # the list of the effective emissions (No node_em)
        self.effective_em_mix # the list of effective emission mixtures (No node_em)
        self.effective_tr_pos # dictionary [statename] number of effective transition 
        self.effective_em_pos # dictionary [statename] number of effective emissions (No node_em)
        self.hmm_tr # precalculated transitions
        self.em_updatable = intersection({not tied}, {not fixed}) 
        self.em_mix_updatable = intersection({not tied}, {not fixed}) at mixture level
        self.tr_updatable = intersection({not tied}, {not fixed})
        # 
        self.labelMusk # dictionary of labels and musk values
        self.mA # fully connected matrix version of transitions
        
        self.mE # fully connected matrix version of emissions
        --> this needs to be changed to incorporate GMMs!!!
        self.is_mE_used = boolean indicator to determine if the emission probabilities have 
        					been pre-computed, only for discrete symbol emissions.
        self.mixtures = Yes/No indicator for using GMM emissions or not
        self.dimProfile = dimension of the input observation vector/profile
		                                                                 
    '''
    #__metaclass__ = psyco.compacttype
    def __init__(self,states,emission_alphabet,fix_tr,fix_em,fix_em_mix,mixtures="No",dimProfile=0):
        ''' __init__(self,nstates,states,emission_alphabet) '''
        self.num_states=len(states) # number of states
        self.states=states # the states
        self.emission_alphabet=emission_alphabet 
        # create list of states
        self.state_names=[] # the state names
        for i in range(self.num_states):
           self.state_names.append(states[i].name)
        # create the list of the state which have learnable
        # transitions and emissions
        self.fix_tr=[None]*self.num_states
        self.fix_em=[None]*self.num_states
        self.fix_em_mix=[None]*self.num_states
        for i in range(self.num_states):
            self.fix_tr[i]=fix_tr[self.state_names[i]]  # fix_tr is a dictionary (key,value) = (state_name,Yes/None)
            self.fix_em[i]=fix_em[self.state_names[i]]  # fix_em has the same format as fix_tr
            self.fix_em_mix[i] = fix_em_mix[self.state_names[i]]
        import Sort_HMM
        # topological order of the states with the corresponding null and emitting states 
        (all_links,self.topo_order,self.emits,self.nulls)=Sort_HMM.tolpological_sort(self.state_names,self.states)
        # list of the inlinks and outlinks for each states
        # also restricted to nulls and emitting
        (self.out_s,self.in_s,self.out_s_e,self.out_s_n,self.in_s_e,self.in_s_n)=Sort_HMM.make_links(self.emits,self.nulls,all_links)
        # end states, all, nulls and emitting
        (self.end_s,self.end_s_e,self.end_s_n)=Sort_HMM.make_ends(self.states,self.emits,self.nulls)
        # now we eliminate 0 from null states Begin must be a null state!!
        i=self.nulls.index(0)
        self.nulls=self.nulls[0:i]+self.nulls[i+1:]
        self.label_list=[]
        for s in self.states:
            if (s.label != 'None') and (s.label not in self.label_list): 
                self.label_list.append(s.label)
        self.effective_tr=[] # the list of the effective transitions (No node_tr)
        self.effective_em=[] # the list of the effective emissions (No node_em)
        self.effective_em_mix = [] # the list of effective emission mixtures (No node_em)
        self.effective_tr_pos={} # dictionary [statename] number of effective transition 
        self.effective_em_pos={} # dictionary [statename] number of effective emissions (No node_em)
        self.effective_em_mix_pos = {} # dictionary [statename] number of effective emission mixtures (No node_em)
        self.em_updatable = [] # these are used when we update the parameters set_param 
        self.em_mix_updatable = [] # these used when we update the parameters gmm_set_param
        self.tr_updatable = [] # these are used when we update the parameters set_param
        for i in range(self.num_states):
            name=self.states[i].get_tr_name()
            if name not in self.effective_tr:
                self.effective_tr.append(name)
                if not self.fix_tr[i]:
                    self.tr_updatable.append(i)
            name=self.states[i].get_em_name()
            if i in self.emits and name not in self.effective_em:
                self.effective_em.append(name)
                if not self.fix_em[i]:
                    self.em_updatable.append(i)
            if i in self.emits:
                gmm_name = self.states[i].get_em_mix_name()
                if i in self.emits and (gmm_name not in self.effective_em_mix):
                    self.effective_em_mix.append(gmm_name)
                    if not self.fix_em_mix[i]:
                        self.em_mix_updatable.append(i)
        for s in self.topo_order:
            self.effective_tr_pos[s]=self.effective_tr.index(self.states[s].get_tr_name())
            if s in self.emits:
                self.effective_em_pos[s]=self.effective_em.index(self.states[s].get_em_name())
                self.effective_em_mix_pos[s] = self.effective_em_mix.index(self.states[s].get_em_mix_name())

        # precalculate transitions  
        self.hmm_tr={} # precalculated transitions
        self.hmm_ln_tr={} # precalculated transitions
        for s in self.topo_order:
            for t in self.topo_order:
                self.hmm_tr[(s,t)]=self.states[s].a(self.states[t])
                self.hmm_ln_tr[(s,t)]= self.states[s].ln_a(self.states[t]) 
       
       # The self.mE pre-computed matrix is only set when working with discrete symbols.
       # Otherwise one has to compute the emission probabilities when provided with the 
       # observable emission vector.
        if (self.states[0].get_type_name() == "node_em"):
        	self.mE=NUM.array([[0.0]*len(self.emission_alphabet)]*self.num_states,ARRAYFLOAT)
        	print "trying to set_mE()\n"
        	self.set_mE()
        	self.is_mE_used = True
        else:
            self.is_mE_used = False
       
        self.mA=NUM.array([[0.0]*self.num_states]*self.num_states,ARRAYFLOAT)
        self.ln_mA=NUM.array([[self.__safe_log(0.0)]*self.num_states]*self.num_states,ARRAYFLOAT)
        self.set_mA()
        self.set_ln_mA()
        
        self.set_labelMusk()
        self.mixtures = mixtures  #Yes/No indicator for using GMM emissions
        self.dimProfile = dimProfile  # dimension of observation vectors/profiles
                 
    def a(self,i,j):
        ''' a(self,i,j) transition probability between i an j '''
        return self.hmm_tr[(i,j)]
#        return self.states[i].a(self.states[j])

    def ln_a(self,i,j):
        ''' ln_a(self,i,j) log of the transition probability between i an j '''
        return self.hmm_ln_tr[(i,j)]
#        return self.states[i].ln_a(self.states[j])

    def set_a(self,i,j,val):
        ''' set_a(self,i,j,val) sets the transition probability between i an j '''
        self.states[i].set_a(self.states[j],val)
        self.hmm_tr[(i,j)]=val
        self.hmm_ln_tr[(i,j)]=self.states[i].ln_a(self.states[j])
        return 

    def e(self,i,x):
        ''' e(self,i,x) emission probability of x in the state i '''
        return self.states[i].e(x)

    def ln_e(self,i,x):
        ''' ln_e(self,i,x) log of the emission probability of x in the state i '''
        return self.states[i].ln_e(x)
    
    def set_e(self,i,x,val):
	''' set_e(self,i,x,val) sets with val the emission probability of x in the state i '''
	return self.states[i].set_e(x,val)
    
    # use the following member functions when working with GMMs
    def set_mix_num(self,state,value):
    	'''
    		sets state's mixture number with value
    	'''
    	return self.states[state]._node_em.set_mix_num(value)
    
    def set_mix_weight(self,state,i,weight):
    	'''
    		set state mixture i's weight value
    	'''
    	return self.states[state]._node_em.set_mix_weight(i,weight)
    
    def normalise_mix_weights(self,state):
	'''
	ensures that the sum of the mixture weights in a particular state 
	is one.
	'''
	return self.states[state]._node_em.normalise_mix_weights()
    
    def set_mix_density(self,state,i,mean,covariance,precision):
        '''
            set state mixture i's density function
        '''
        tmp_name = self.states[state]._node_em._mix_densities[i].get_name()
        tmp_density = State.mixture_density(tmp_name,mean,covariance,precision)
        self.states[state]._node_em.set_mix_density(i,tmp_density)
    
    def save(self,file):
        ''' save the hmm into cPickle file '''
        import HMM_IO
        HMM_IO.save_cPickle(self,file) 

    def set_mA(self):
        ''' set_mA(self) set sel.mA the fully connected matrix version of the transitions''' 
        for s in self.topo_order:
            for t in self.topo_order:
                self.mA[s][t]=self.a(s,t)
        self.set_ln_mA()

    def set_ln_mA(self):
        for s in self.topo_order:
            for t in self.topo_order:
                self.ln_mA[s][t]=self.ln_a(s,t)
    
    def set_mE(self):
        ''' set_mA(self) set sel.mA the fully connected matrix version of the emissions''' 
        # this is only used and initialized when using the discrete symbol version. 
        # NOT the GMM version.
        if self.is_mE_used:        
            for s in self.topo_order:
                i=0 
                for c in self.emission_alphabet:
                    self.mE[s][i]=self.e(s,c)
                    i+=1
        else:
            print "GMM do not use a precomputed emissions matrix."
                
    def set_labelMusk(self):
        '''  set_labelMusk(self) set a dictionary of label and allow state values'''
        self.labelMusk={} 
        for lab in self.label_list:
            self.labelMusk[lab]=NUM.array([0.0]*self.num_states,ARRAYFLOAT)
            for s in self.topo_order:
                if self.states[s].label == lab :
                    self.labelMusk[lab][s]=1.0

    def load_file(self,file):
        ''' load the hmm from file '''
        import HMM_IO
        obj=HMM_IO.load_cPickle(file)
        self.__dict__=obj.__dict__

    def write_for_humans(self,filename):
        ''' write_for_humans(self,filename)
           write on filename the self in the same format
           it is read by parse_text
        '''
        import HMM_IO
        HMM_IO.write_for_humans(self,filename)

        
    def randomWalk(self,maxLen):
        '''
           randomWalk(self,maxLen)
           rerutn a list containing tuples of label emission
                  of maximal length = maxLen
                  In reality if not all the states can be ending states
                  the list length can be longer or shorter!
        '''
        from random import randint
        stateNum=0
        state=self.states[0]
        name=state.name
        retlist=[]
        i=0
        while i<maxLen and  self.out_s[stateNum] :
            nextPos,name=_getRand(randint,state.get_transitions(),state.out_links)
            stateNum=self.state_names.index(name)
            state=self.states[stateNum]
            em_letters=state.em_letters
            if em_letters != [] : # not null  
                nextPos,emiss=_getRand(randint,state.get_emissions(),em_letters)
                label=state.label
                retlist.append((name,label,emiss))
                i=i+1
        while not state.end_state :
            nextPos,name=_getRand(randint,state.get_transitions(),state.out_links)
            stateNum=self.state_names.index(name)
            state=self.states[stateNum]
            em_letters=state.em_letters
            if em_letters != [] : # not null
                nextPos,emiss=_getRand(randint,state.get_emissions(),em_letters)
                label=state.label
                retlist.append((name,label,emiss))
        return retlist
    
    def __safe_log(self,x):
        if x < DEF.small_positive:
            return DEF.big_negative
        else:
            return NUM.log(x)
        
    def randomWalkGMM(self,maxLen):
        '''
           randomWalk(self,maxLen)
           rerutn a list containing tuples (observation,label,hidden state)
                  In reality if not all the states can be ending states
                  the list length can be longer or shorter!
        
            For a thorough reference please refer to the Sklearn source code (github account) as 
            the below implementation follows their code rather closely. 
            Thank you SkLearn team!!! 
        '''
        
        # BEGIN and recursive state stages
        from random import randint
        stateNum=0
        state=self.states[0]
        name=state.name
        retlist=[]
        i=0
        while i<maxLen and  self.out_s[stateNum] :
            nextPos,name=_getRand(randint,state.get_transitions(),state.out_links)
            stateNum=self.state_names.index(name)
            state=self.states[stateNum]
            
            # generate sample observation based on GMM in current state
            # need to update this to include null/silent states!!!
            state_gmm = state.get_emissions() # returns GMM object for current state
            if state_gmm.get_mix_num() > 0: # must be an emitting state
                n_samples = 1
                X = NUM.empty((n_samples,len(state_gmm.get_mixture_density(0).get_mean())))
                num_mix = state_gmm.get_mix_num()
                weight_cdf = NUM.cumsum([state_gmm.get_mix_weight(m) for m in range(num_mix)])
                # generate one random sample state, used to choose which mixture component to use. 
                # Assessed by component weights
                rand = NUM.random.mtrand._rand.rand(n_samples) 
                # decide which component to use for each sample
                comps = weight_cdf.searchsorted(rand)
                assert comps <= num_mix # mixture component check!!!
                # for each component, generate all needed samples
                for comp in range(num_mix):
                    # occurrences of current component in X
                    comp_in_X = (comp == comps)
                    num_comp_in_X = comp_in_X.sum()
                    if num_comp_in_X > 0:
                        X[comp_in_X] = _sample_gaussian(state_gmm.get_mixture_density(comp).get_mean(), 
                                                    state_gmm.get_mixture_density(comp).get_cov(),
                                                    num_comp_in_X).T
            
                label = state.label
                retlist.append((X,label,name))
                i += 1
        
        # Ending state stages 
        while not state.end_state :
            nextPos,name=_getRand(randint,state.get_transitions(),state.out_links)
            stateNum=self.state_names.index(name)
            state=self.states[stateNum]
            
            # generate sample observation based on GMM in current state
            # need to update this to include null/silent states!!!
            state_gmm = state.get_emissions() # returns GMM object for current state
            if state_gmm.get_mix_num() > 0: # must be an emitting state
                n_samples = 1
                X = NUM.empty((n_samples, len(state_gmm.get_mixture_density(0).get_mean())))
                num_mix = state_gmm.get_mix_num()
                weight_cdf = NUM.cumsum([state_gmm.get_mix_weight(m) for m in range(num_mix)])
                # generate one random sample state, used to choose which mixture component to use. 
                # Assessed by component weights
                rand = NUM.random.mtrand._rand.rand(n_samples) 
                # decide which component to use for each sample
                comps = weight_cdf.searchsorted(rand)
                # for each component, generate all needed samples
                for comp in range(num_mix):
                    # occurrences of current component in X
                    comp_in_X = (comp == comps)
                    num_comp_in_X = comp_in_X.sum()
                    if num_comp_in_X > 0:
                        X[comp_in_X] = _sample_gaussian(state_gmm.get_mixture_density(comp).get_mean(), 
                                                    state_gmm.get_mixture_density(comp).get_cov(),
                                                    num_comp_in_X).T
            
                label = state.label
                retlist.append((X,label,name))
            
        return retlist
###########
