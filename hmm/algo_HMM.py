'''
This file contains the main algorithms
used to train the hmms, namely forward,backward,viterbi and baum-welch

TODO: tests
'''

try:
	import psyco
	psyco.full()
except:
	pass
# Usually we use nicknames for the states and links
#
# B = {Begin}, E= {all emitting states}, N={all null States}
# S ={ all states} = B U E U N
#
# O[t] = {s in S | a(t,s) is allowed } // outlinks
# I[t] = {s in S | a(s,t) is allowed } // inlinks
#
# OE[t] = O[t] restricted to all emitting states
# ON[t] = O[t] restricted to all null states (not B)
#
# O[t]= OE[t] U ON[t] 
# I[t]= IE[t] U IN[t] 
# ENDS = list of the states thac can end
# 
# usually s is the current state and t is the inlinked (outlinked) state
#
#
#                     States ......
#                 B {       S       } 
#     Sequence    B |   E    |  N   |
#                 0 |1 .....m|m+1..n|  
#            0                      |
#            1                      |
#            2                      |
#            ..                     |
#            ..                     |
#            L                      |
#
# N[0] < N[1] ...
#
# the sequence of symbols (vectors) is of length L and 
# ranges from 0 to L-1. So there is a difference of 1 with respect
# to the equation. This means that fo the forward/backward/viterbi etc.
# i-position the sequence  


#import hmm
import sys
from Def import DEF
import numpy as NUM
import copy
import nearPD_Higham
from sklearn.cluster import KMeans as sk_kmeans
from sklearn.cluster import SpectralClustering as sk_spectralclust
from sklearn.covariance import GraphLassoCV, graph_lasso

ARRAYFLOAT=NUM.float64
ARRAYINT=NUM.int


def __safe_log(x):
	if x < DEF.small_positive:
		return DEF.big_negative
	else:
		return NUM.log(x)

def __safe_exp(x):
	if x <= DEF.big_negative:
		return 0
	else:
		return NUM.exp(x)

def __e_ln_sum(elnx,elny):
	'''
		computes the safe log of a sum ln(x+y) with the individual ln(x) and ln(y) as inputs.
		See the paper "Numerically Stable Hidden Markov Model Implementation" by T. Mann.
	'''
	if (elnx == DEF.big_negative) or (elny == DEF.big_negative):
		if elnx == DEF.big_negative:
			result = elny
		else:
			result = elnx
	else:
		if (elnx > elny):
			result = elnx + __safe_log(1+ NUM.exp(elny - elnx))
		else:
			result = elny + __safe_log(1+ NUM.exp(elnx - elny))
	
	return result

def for_back_mat(hmm, seq, Scale=None, labels=None):
	''' forward/backward algorithm 
		for_back_mat(hmm, seq, Scale=None, labels=None):
		-> 
			return(for_matrix,back_matrix,eMat,scale,log_Prob))
			for_matrix  = calculated forward matrix shape=(Seqence length+1, number of states)
			back_matrix = calculated backward matrix shape=(Seqence length+1, number of states)
			eMat        = precalculated emission probability matrix shape=(number of states,Seqence length)
			scale       = scale factor array shape=(Seqence length+1,) # if Scale!=None
			log_Prob    = log( P(sequence | hmm) )  
			              the states are in the hmm.topo_order
	'''
	eMat=eval_eMatLab(hmm, seq,labels)
	(f,lp,Scale,end_scale)=_forward_mat_no_null(hmm, seq, eMat, Scale, labels)
	#(f,lp,Scale,end_scale)=_forward_mat(hmm, seq, eMat, Scale, labels)
	b=_backward_mat_no_null(hmm, seq, eMat, Scale, end_scale, labels)
	#b=_backward_mat(hmm, seq, eMat, Scale, end_scale, labels)
	return(f,b,eMat,Scale,lp)

def ln_for_back_mat(hmm, seq, labels=None):
	''' forward/backward algorithm computed in the log transform space
		ln_for_back_mat(hmm, seq, labels=None):
		-> 
			return(for_matrix,back_matrix,eMat,scale,log_Prob))
			for_matrix  = calculated forward matrix shape=(Seqence length+1, number of states)
			back_matrix = calculated backward matrix shape=(Seqence length+1, number of states)
			eMat        = precalculated emission probability matrix shape=(number of states,Seqence length)
			scale       = scale factor array shape=(Seqence length+1,) # if Scale!=None
			log_Prob    = log( P(sequence | hmm) )  
			              the states are in the hmm.topo_order
	'''
	ln_eMat=ln_eval_eMatLab(hmm, seq,labels)
	(ln_f,lp)=_ln_forward_mat_no_null(hmm, seq, ln_eMat, labels)
	ln_b=_ln_backward_mat_no_null(hmm, seq, ln_eMat,labels)
	return(ln_f,ln_b,ln_eMat,lp)


def seq_log_Prob(hmm, seq, Scale=None, labels=None):
	''' forward algorithm to compute the log_Prob of the sequence
		seq_log_Prob(hmm, seq, Scale=None, labels=None):
		-> 
			return
			log_Prob    = log( P(sequence | hmm) )  
			              the states are in the hmm.topo_order
	'''
	eMat=eval_eMatLab(hmm, seq,labels)
	(f,lp,Scale,end_scale)=_forward_mat(hmm, seq, eMat, Scale, labels)
	return lp

####################################

def eval_eMatLab(hmm, seq,labels=None):
	''' compute the eMat 
		eval_eMatLab(hmm, seq,labels)
		eMat[s][i] = e(s,seq[i])\delta_{label[i],label_states} is the precalculated emission probability matrix
        
		Don't really need to have the if conditional, as the node_em class has been used as a base class to 
		provide polymorphic functionality
	'''
	L=len(seq)
	E=hmm.emits # the list of the emitting states
	if(type(seq[0])!=type('String')): # this means we are dealing with vectors
		'''        
		eMat= NUM.dot(hmm.mE,NUM.array(seq).T) # remember seq[i-1] is x_i      
		if labels:
			eMat=eMat.T
			for i in range(L):
				if labels[i]:
					eMat[i]*=hmm.labelMusk[labels[i]]
			eMat=eMat.T        
		'''
		eMat=NUM.array([[0.0]*len(seq)]*len(hmm.topo_order),ARRAYFLOAT) # setup array for emissions for this particular protein sequence
		for i in range(L):
			for s in E:  # for each sequence position and each emitting state we update the probability emission matrix        
				if (not labels or not labels[i-1] or hmm.states[s].label == labels[i]):
					eMat[s][i] = hmm.e(s,seq[i]) # will use hmm.state.node_em_gmm to compute the emission of the current observation vector in state s    
	else: # only one symbol
		e=hmm.e
		eMat=NUM.array([[0.0]*len(seq)]*len(hmm.topo_order),ARRAYFLOAT)
		for i in range(L):
			for s in E:
				if (not labels or not labels[i-1] or hmm.states[s].label == labels[i]):
					eMat[s][i]= e(s,seq[i]) # remember seq[i-1] is x_i      
	return eMat

def ln_eval_eMatLab(hmm, seq,labels=None):
	''' compute the eMat using log transform space 
		ln_eval_eMatLab(hmm, seq,labels)
		ln_eMat[s][i] = ln_e(s,seq[i])\delta_{label[i],label_states} is the precalculated emission probability matrix
        
		Don't really need to have the if conditional, as the node_em class has been used as a base class to 
		provide polymorphic functionality
	'''
	L=len(seq)
	E=hmm.emits # the list of the emitting states
	if(type(seq[0])!=type('String')): # this means we are dealing with vectors
		'''        
		eMat= NUM.dot(hmm.mE,NUM.array(seq).T) # remember seq[i-1] is x_i      
		if labels:
			eMat=eMat.T
			for i in range(L):
				if labels[i]:
					eMat[i]*=hmm.labelMusk[labels[i]]
			eMat=eMat.T        
		'''
		ln_eMat=NUM.array([[__safe_log(0.0)]*len(seq)]*len(hmm.topo_order),ARRAYFLOAT) # setup array for emissions for this particular protein sequence
		for i in range(L):
			for s in E:  # for each sequence position and each emitting state we update the probability emission matrix        
				if (not labels or not labels[i-1] or hmm.states[s].label == labels[i]):
					ln_eMat[s][i] = hmm.ln_e(s,seq[i]) # will use hmm.state.node_em_gmm to compute the emission of the current observation vector in state s    
	else: # only one symbol
		ln_e=hmm.ln_e
		ln_eMat=NUM.array([[__safe_log(0.0)]*len(seq)]*len(hmm.topo_order),ARRAYFLOAT)
		for i in range(L):
			for s in E:
				if (not labels or not labels[i-1] or hmm.states[s].label == labels[i]):
					ln_eMat[s][i]= ln_e(s,seq[i]) # remember seq[i-1] is x_i      
	return ln_eMat

def _forward_mat_no_null(hmm, seq, eMat, Scale=None, labels=None):
	''' forward algorithm which takes advantage of the precalculated emissions eMat
		_forward_mat_no_null(hmm, seq, eMat, scale=None, labels=None)
		       WARNING: 
		  assumes there are not silent/null states and there are a specific begin and end (null) state
		-> 
			return(for_matrix,log_Prob,scale,end_scale)
			for_matrix  = calculated forward matrix shape=(Seqence length+1, number of states)
			log_Prob    = log( P(sequence | hmm) )  
			scale       = scale factor array shape=(Seqence length+1,) # if Scale!=None
			end_scale   = the normalization factor for the end state
			              if scale is not defined end_scale = 1.0 
			        the states are in the hmm.topo_order
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	# 	since f, b and Scale start with dummy position.
	# 	This imply that for position i we use seq[i-1] and label[i-1]
	#	being this two list shorter 
	L=len(seq) + 1 
	f=NUM.array([[0.0]*hmm.num_states]*L,ARRAYFLOAT)
	if Scale != None:
		Scale=NUM.array([1.0]*L,ARRAYFLOAT)
#    else:
#        Scale=[]
	end_scale=1.0 # the end scale factor, this is needed for the backward
	a=hmm.a # set the transition probabilities
	S=hmm.topo_order # the list of all state including B
	IS=hmm.in_s # inlinks 
	IE=hmm.in_s_e # inlinks from emittings only
	IN=hmm.in_s_n # inlinks from nulls only
	ENDS=hmm.end_s # end states
	B=0 # begin state
	P=0.0 # P(seq) 
	###### START PHASE
	f[0][B]=1.0 
	###### RECURRENCE PHASE for 1 to L-1
	for i in range(1,L):
		# S -> E
		f[i]=NUM.dot(f[i-1],hmm.mA)*eMat.T[i-1]  
		#numpy.arrays use * for element-wise multiplication and dot() for matrix multiplication
		# if we use the scale factor
		if(Scale != None):
			Scale[i]=NUM.sum(f[i]) +DEF.small_positive
			if Scale[i] > 0:
				f[i]/=Scale[i]
			else:
				sys.stderr.write('Error in '+str(i)+'\n')
				print 'Error in computing forward matrix scaling '+str(i)+'\n'
				sys.exit(-1)
    ###### END PHASE 
	for s in hmm.end_s_n:
		for t in hmm.emits:
			f[L-1][s]+=a(t,s)*f[L-1][t]
	for s in ENDS:
		P+=f[L-1][s] # lst index is L-1 [0,L-1]
	# null end states
	if(Scale != None):
		end_scale=P
		lP=__safe_log(P)+ NUM.sum(NUM.log(Scale))
	else:
		lP=__safe_log(P)
		
	return(f,lP,Scale,end_scale)

def _ln_forward_mat_no_null(hmm, seq, ln_eMat, labels=None):
	''' forward algorithm which takes advantage of the precalculated emissions ln_eMat
		_ln-_forward_mat_no_null(hmm, seq, ln_eMat, labels=None)
		       WARNING: 
		  assumes there are not silent/null states and there are a specific begin and end (null) state
		-> 
			return(ln_for_matrix,log_Prob)
			ln_for_matrix  = calculated forward matrix shape=(Seqence length+1, number of states) in log transform space
			log_Prob    = log( P(sequence | hmm) )  
			        the states are in the hmm.topo_order
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	# 	since f, b and Scale start with dummy position.
	# 	This imply that for position i we use seq[i-1] and label[i-1]
	#	being this two list shorter 
	L=len(seq) + 1 
	ln_f=NUM.array([[__safe_log(0.0)]*hmm.num_states]*L,ARRAYFLOAT)
	ln_a=hmm.ln_a # set the transition probabilities
	S=hmm.topo_order # the list of all state including B
	IS=hmm.in_s # inlinks 
	IE=hmm.in_s_e # inlinks from emittings only
	IN=hmm.in_s_n # inlinks from nulls only
	ENDS=hmm.end_s # end states
	B=0 # begin state
	ln_P=__safe_log(0.0) # P(seq) 
	###### START PHASE
	ln_f[0][B]=__safe_log(1.0) 
	###### RECURRENCE PHASE for 1 to L-1
	for t in range(1,L):
		# S -> E
		#f[i]=NUM.dot(f[i-1],hmm.mA)*eMat.T[i-1]  
		#numpy.arrays use * for element-wise multiplication and dot() for matrix multiplication
		for j in range(hmm.num_states):
			ln_alpha = __safe_log(0.0)
			for i in range(hmm.num_states): 
				ln_prod = ln_f[t-1][i] + ln_a(i,j)
				ln_alpha = __e_ln_sum(ln_alpha,ln_prod)
			ln_f[t][j] = ln_alpha + ln_eMat[j][t-1]
    
    ###### END PHASE 
	for s in ENDS:
		ln_P = __e_ln_sum(ln_P,ln_f[L-1][s]) # lst index is L-1 [0,L-1]
		
	return(ln_f,ln_P)

######################################

def _forward_mat(hmm, seq, eMat, Scale=None, labels=None):
	''' forward algorithm which takes advantage of the precalculated emissions eMat
		_forward_mat(hmm, seq, eMat, scale=None, labels=None)
		-> 
			return(for_matrix,log_Prob,scale,end_scale)
			for_matrix  = calculated forward matrix shape=(Seqence length+1, number of states)
			log_Prob    = log( P(sequence | hmm) )  
			scale       = scale factor array shape=(Seqence length+1,) # if Scale!=None
			end_scale   = the normalization factor for the end state
			              if scale is not defined end_scale = 1.0 
				        the states are in the hmm.topo_order
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	# 	since f, b and Scale start with dummy position.
	# 	This imply that for position i we use seq[i-1] and label[i-1]
	#	being this two list shorter 
	L=len(seq) + 1 
	f=NUM.array([[0.0]*hmm.num_states]*L,ARRAYFLOAT)
	
	if Scale != None:
		Scale=NUM.array([1.0]*L,ARRAYFLOAT)
#    else:
#        Scale=[]
	end_scale=1.0 # the end scale factor, this is needed for the backward
	a=hmm.a # set the transition probabilities
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	S=hmm.topo_order # the list of all state including B
	IS=hmm.in_s # inlinks 
	IE=hmm.in_s_e # inlinks from emittings only
	IN=hmm.in_s_n # inlinks from nulls only
	ENDS=hmm.end_s # end states
	B=0 # begin state
	P=0.0 # P(seq) 
	###### START PHASE
	f[0][B]=1.0 
	for s in E: 
		f[0][s]=0.0
	# from B -> N 
	for s in N:
		f[0][s]=a(B,s)
	# from N -> N
	for s in N:
		for t in IN[s]: 
			f[0][s]+=f[0][t]*a(t,s)
	# if we use the scale factor
	if(Scale != None):
		Scale[0]=NUM.sum(f[0])
		f[0]/=Scale[0]
	
	print "forward beginning recurrence phase"	
	###### RECURRENCE PHASE for 1 to L-1
	for i in range(1,L):
		# S -> E
		for s in E:
			# update done only if labels are not used
			# or labels are free == None
			# or labels corresponds (labels[i-1] correspond position i)
			for t in IS[s]:
				f[i][s]+=a(t,s)*f[i-1][t]
				print "pos %d ,\n a(%d,%d) = "%(i,t,s),a(t,s)
				print "f[%d][%d] = "%(i-1,t),f[i-1][t] 
				print "update result f[%d][%d] = "%(i,s),f[i][s]
			#print "before recurr-emission f[%d][%d] = "%(i,s),f[i][s]
			f[i][s]*= eMat[s][i-1] # remember seq[i-1] is x_i 
			#print "after recurr-emission f[%d][%d] = "%(i,s),f[i][s]
			print "eMat[%d][%d] = "%(s,i-1),eMat[s][i-1]	
		# E -> N . Note the i-level is the same
		for s in N:
			for t in IE[s]:
				f[i][s]+=a(t,s)*f[i][t]
		# N -> N . Note the i-level is the same
		for s in N:
			for t in IN[s]:
				f[i][s]+=a(t,s)*f[i][t]
		# if we use the scale factor
		if(Scale != None):
			Scale[i]=NUM.sum(f[i])+DEF.small_positive
			if Scale[i] > 0:
				f[i]/=Scale[i]
			else:
				sys.stderr.write('Error in '+str(i)+' Scale should be non-zero\n')
				sys.exit(-1)
				
	###### END PHASE 
	for s in ENDS:
		P+=f[L-1][s] # lst index is L-1 [0,L-1]
	if(Scale != None):
		end_scale=P
		lP=__safe_log(P)+ NUM.sum(NUM.log(Scale))
	else:
		lP=__safe_log(P)
		
	return(f,lP,Scale,end_scale)
######################################

def _backward_mat_no_null(hmm, seq, eMat, Scale=None, end_scale=1.0, labels=None):
	''' backward algorithm which takes advantage of the precalculated emission probabilities eMat 
		This function must not be called before _forward
		_backward_mat_no_null(hmm, seq, eMat, Scale=None, end_scale, labels=None)
		WARNING: 
			assumes there are not silent/null states and there are a specific begin and end (null) state
		-> returns (back_matrix) 
			back_matrix = calculated backward matrix shape=(Seqence length+1, number of states)
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	#   since f, b and Scale start with dummy position.
	#   This imply that for position i we use seq[i-1] and label[i-1]
	#   being this two list shorter 
	L=len(seq)+1
	b=NUM.array([[0.0]*hmm.num_states]*L,ARRAYFLOAT)
	a=hmm.a # set the transition probabilities
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	S=hmm.topo_order # the list of all state including B
	ENDS=hmm.end_s # end states
	ENDE=hmm.end_s_e # emitting end states
	ENDN=hmm.end_s_n # null end states
	OS=hmm.out_s # outlinks
	OE=hmm.out_s_e # outlinks from emittings only
	ON=hmm.out_s_n # outlinks from nulls only
	B=0 # begin state
	###### START PHASE
	for s in ENDS:
		b[L-1][s]=1.0/end_scale
	# E <- N 
	for t in ENDN:
		for s in E:
			b[L-1][s]=b[L-1][t]*a(s,t)
			
	if(Scale!=None):
		b[L-1]/=Scale[L-1]
	###### RECURRENCE PHASE for L-2 to 1
	for i in range(L-2,0,-1):
		b[i]=NUM.dot(hmm.mA,b[i+1]*eMat.T[i]) # component-wise multiplication between b and eMat and matrix multiplication with hmm.mA
		if(Scale != None):
			b[i]/=Scale[i]
			
	###### END PHASE
	# BEGIN
	for t in OE[B]:
		b[0][B]+=a(B,t)*b[1][t]*eMat[t][0] # seq[0] is the first position !!
	if(Scale != None):
		b[0]/=Scale[0]
		
	return(b)

######################################

def _ln_backward_mat_no_null(hmm, seq, ln_eMat, labels=None):
	''' backward algorithm which takes advantage of the precalculated emission probabilities ln_eMat 
		This function must not be called before _ln_forward
		_ln_backward_mat_no_null(hmm, seq, eMat, labels=None)
		WARNING: 
			assumes there are not silent/null states and there are a specific begin and end (null) state
		-> returns (back_matrix) 
			ln_back_matrix = calculated backward matrix shape=(Seqence length+1, number of states) using log transform space
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	#   since f, b and Scale start with dummy position.
	#   This imply that for position i we use seq[i-1] and label[i-1]
	#   being this two list shorter 
	L=len(seq)+1
	ln_b=NUM.array([[__safe_log(0.0)]*hmm.num_states]*L,ARRAYFLOAT)
	ln_a=hmm.ln_a # set the transition probabilities
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	S=hmm.topo_order # the list of all state including B
	ENDS=hmm.end_s # end states
	ENDE=hmm.end_s_e # emitting end states
	ENDN=hmm.end_s_n # null end states
	OS=hmm.out_s # outlinks
	OE=hmm.out_s_e # outlinks from emittings only
	ON=hmm.out_s_n # outlinks from nulls only
	B=0 # begin state
	###### START PHASE
	for s in ENDS:
		ln_b[L-1][s]= __safe_log(1.0)
	
	###### RECURRENCE PHASE for L-2 to 1
	for t in range(L-2,0,-1):
		for i in range(hmm.num_states):
			ln_beta = __safe_log(0.0)
			for j in range(hmm.num_states):
				ln_prod_1 = ln_b[t+1][j] + ln_eMat[j][t]
				ln_prod_2 = ln_a(i,j) + ln_prod_1
				ln_beta = __e_ln_sum(ln_beta, ln_prod_2)
			ln_b[t][i] = ln_beta
		
	###### END PHASE
	# BEGIN
	for t in OE[B]:
		ln_prod = ln_a(B,t) + ln_b[1][t] + ln_eMat[t][0] # seq[0] is the first position !!
		ln_b[0][B] = __e_ln_sum(ln_b[0][B], ln_prod) 
		
	return ln_b

######################################


def _backward_mat(hmm, seq, eMat, Scale=None, end_scale=1.0, labels=None):
	''' backward algorithm which takes advantage of the precalculated emission probabilities eMat 
		This function should not be called before _forward
		_backward_mat(hmm, seq, eMat, Scale=None, end_scale, labels=None)
		-> returns (back_matrix) 
		back_matrix = calculated backward matrix shape=(Seqence length+1, number of states)
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	#   since f, b and Scale start with dummy position.
	#   This imply that for position i we use seq[i-1] and label[i-1]
	#   being this two list shorter 
	L=len(seq)+1
	b=NUM.array([[0.0]*hmm.num_states]*L,ARRAYFLOAT)
	a=hmm.a # set the transition probabilities
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	RN=copy.deepcopy(hmm.nulls)
	RN.reverse() # the list of the silent state in reversed order
	S=hmm.topo_order # the list of all state including B
	ENDS=hmm.end_s # end states
	ENDE=hmm.end_s_e # emitting end states
	ENDN=hmm.end_s_n # null end states
	OS=hmm.out_s # outlinks
	OE=hmm.out_s_e # outlinks from emittings only
	ON=hmm.out_s_n # outlinks from nulls only
	B=0 # begin state
	###### START PHASE
	for s in ENDN:
		b[L-1][s]=1.0/end_scale
	for s in ENDE:
		# update done only if labels are not used
		# or labels are free == None
		# or labels corresponds
		# Remember labels start from 0 and it is shorter !
		if(not labels or not labels[L-2] or hmm.states[s].label == labels[L-2]):
			b[L-1][s]=1.0/end_scale
	# N <- N in reverse order
	for s in RN:
		if s not in ENDN:
			for t in ON[s]:
				b[L-1][s]+=b[L-1][t]*a(s,t)
	# E <- N 
	for s in E:
		# update done only if labels are not used
		# or labels are free == None
		# or labels corresponds
		# Remember labels start from 0 and it is shorter !
		if (s not in ENDE) and (not labels or not labels[L-2] or hmm.states[s].label == labels[L-2]):
			for t in ON[s]:
				b[L-1][s]+=b[L-1][t]*a(s,t)
	if(Scale!=None):
		b[L-1]/=Scale[L-1]
		
	###### RECURRENCE PHASE for L-2 to 1
	for i in range(L-2,0,-1):
		# N(i) <- E(i+1)
		for s in N:
			for t in OE[s]:
				b[i][s]+=a(s,t)*b[i+1][t]*eMat[t][i] # seq[i] is postion i+1!!
		# N(i) <- N(i) in reverse order
		for s in RN:
			for t in ON[s]:
				b[i][s]+=b[i][t]*a(s,t)
		for s in E:
			# labels[i-1] is position i !!
			# E(i) <- E(i+1)
			for t in OE[s]:
				b[i][s]+=a(s,t)*b[i+1][t]*eMat[t][i] # seq[i] is postion i+1!!
			# E(i) <- N(i) 
			for t in ON[s]:
				b[i][s]+=b[i][t]*a(s,t)
		if(Scale != None):
			b[i]/=Scale[i]
			
	###### END PHASE 
	for s in RN:
		for t in OE[s]: # N <- E
			b[0][s]+=a(s,t)*b[1][t]
	for s in RN:
		for t in ON[s]: # N <- N
			b[0][s]+=a(s,t)*b[1][t]
	# BEGIN
	for t in OE[B]:
		b[0][B]+=a(B,t)*b[1][t]*eMat[t][0] # seq[0] is the first position !!
	for t in ON[B]:
		b[0][B]+=a(B,t)*b[0][t]
	if(Scale != None):
		b[0]/=Scale[0]
		
	return(b)
######################################


######################################
#
# DECODING
#
######################################

def viterbi(hmm, seq, returnLogProb=None, labels=None):
	''' viterbi algorithm 
		viterbi(hmm, seq,  returnLogProb=None, labels=None):
		-> 
		if returnLogProb == None:
			return(best_state_path, best_path_score)
		else:
			return(best_state_path, pest_path_score, best_path_values)
	'''
	if returnLogProb: 
		return _viterbi(hmm, seq, labels)
	else:
		best_state_path, pest_path_score, best_path_values=_viterbi(hmm, seq, labels)
		return best_state_path, pest_path_score
######################################

def viterbi_label(hmm, seq, labels=None):
	''' viterbi algorithm 
		viterbi_label(hmm, seq, labels):
		-> 
			return(best_LABEL_path)
			best_LABEL_path = the path containing the labels of the best states
	'''
	best_LABEL_path=[]
	(best_state_path,val,pathVal)=_viterbi(hmm, seq, labels)
	for s in best_state_path:
#        if s in hmm.emits:
		best_LABEL_path.append(hmm.states[s].label)
	return (best_LABEL_path,val)
######################################

def _viterbi(hmm, seq, labels=None):
	''' viterbi algorithm 
		_viterbi(hmm, seq, labels=None):
		-> 
			return(best_state_path, val, pathVal)
			best_state_path =  the best path
			val = the score of the best path
			pathVal = the list with the log of the emission probabilities for each 
			selected state. If null the value is DEF.big_negative
	'''
	# settings
	# PLEASE NOTE THAT seq and labels have different indices
	# 	since f, b and Scale start with dummy position.
	# 	This imply that for position i we use seq[i-1] and label[i-1]
	#	being this two list shorter 
	L=len(seq)+1
	ln_f=NUM.array([[__safe_log(0.0)]*hmm.num_states]*L,ARRAYFLOAT)
	bkt=NUM.array([[0]*hmm.num_states]*L,ARRAYINT)  # backtrack pointer matrix where integer values represent various states
	ln_a=hmm.ln_a # set the transition probabilities (this is a function pointer)
	if(type(seq[0])!=type('String')): # this means we are dealing with vectors
		#ln_e=hmm.ln_eV
		ln_e = hmm.ln_e  #this is also a function pointer which uses dynamic binding of the hmm.state.node_em object for the GMM
	else: # only one symbol
		ln_e=hmm.ln_e
    
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	S=hmm.topo_order # the list of all state including B
	IS=hmm.in_s # inlinks 
	IE=hmm.in_s_e # inlinks from emittings only
	IN=hmm.in_s_n # inlinks from nulls only
	ENDS=hmm.end_s # end states
	B=0 # begin state
	###### START PHASE
	ln_f[0][B]=__safe_log(1.0) 
	bkt[0][B]=-1
	for s in E: 
		ln_f[0][s]=DEF.big_negative
		bkt[0][s]=B
	# from N+B -> N
	for s in N:
		for t in IN[s]: # IN[s] contains B too  
			if(ln_f[0][s]<ln_a(t,s)+ln_f[0][t]):
				ln_f[0][s]=ln_a(t,s)+ln_f[0][t]  # this is the update formula for the null states see Durbin et al.
				bkt[0][s]=t
	###### RECURRENCE PHASE for 1 to L-1
	for i in range(1,L):
		# S -> E
		for s in E:
			# update done only if labels are not used
			# or labels corresponds (labels[i-1] is position i !!)
			if(not labels or not labels[i-1] or hmm.states[s].label == labels[i-1]):
				for t in IS[s]:
					ltmp=ln_a(t,s)+ln_f[i-1][t]
					if(ln_f[i][s]<ltmp):			
						ln_f[i][s]=ltmp
						bkt[i][s]=t
				ln_f[i][s]+=ln_e(s,seq[i-1]) # seq[i-1] is position i!!
		# E -> N . Note the i-level is the same
		for s in N:
			for t in IE[s]:
				ltmp=ln_a(t,s)+ln_f[i][t]
				if(ln_f[i][s]<ltmp):			
					ln_f[i][s]=ltmp
					bkt[i][s]=t
		# N -> N . Note the i-level is the same
		for s in N:
			for t in IN[s]:
				ltmp=ln_a(t,s)+ln_f[i][t]
				if(ln_f[i][s]<ltmp):			
					ln_f[i][s]=ltmp
					bkt[i][s]=t
	###### END PHASE 
	bestval=DEF.big_negative
	pback=None
	for s in ENDS:
		#print "End state %s, cur bestval, lf[L-1][s]"%(s),bestval,lf[L-1][s]
		#print "pback", pback
		if(bestval<ln_f[L-1][s]):	    
			bestval=ln_f[L-1][s]
			pback=s	
	if not pback:
		print "ERRORRRRR!!!"
	
	#print "new bestval",bestval
	#print "new pback", pback
		
	###### BACKTRACE
	logProbPath=[]
	best_path=[]
	i=L-1
	
	while i >= 0 and pback != B:
		best_path.insert(0,pback)
		ptmp=bkt[i][pback]
		#print "pback",pback
		#print "ptmp",ptmp
		#print "pos i",i	
		
		if pback in hmm.emits:
			logProbPath.insert(0, ln_e(pback,seq[i-1]))
			i-=1
		else:
			logProbPath.insert(0, DEF.big_negative)
		
		pback=ptmp
	
	#print "len of seq and bestpath",len(seq),len(best_path)
	#print "\n end of current seq \n"	
	
	#the last value in the best_path is the end state, this could be an artificial null/silent state	
	return(best_path,bestval,logProbPath)
######################################

def ap_viterbi(hmm, seq, label_list=None, labels=None, Scale='y', returnProbs=None):
	''' viterbi algorithm on the a posteriori
		if label_list == None : the state path is returned
		else :          the label path is returned 
		
		labels sequence labelling # this is used in the forward/backward
		Scale != None we use the scaling  
		-> if returnProbs == None
		     return(best_state_path,val)
		     best_state_path = the best path
		     val = the score of the best path
		else
		     return(best_state_path,val,returnProbs)
		     best_state_path = the best path
		     val = the score of the best path
		returnProbs = list of the Probability for each state in best_state_path  
	'''
	# settings
	L=len(seq)+1
	ln_f=NUM.array([[__safe_log(0.0)]*hmm.num_states]*L,ARRAYFLOAT)
	bkt=NUM.array([[-1]*hmm.num_states]*L,ARRAYINT)
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	S=hmm.topo_order # the list of all state including B
	IS=hmm.in_s # inlinks 
	IE=hmm.in_s_e # inlinks from emittings only
	IN=hmm.in_s_n # inlinks from nulls only
	ENDS=hmm.end_s # end states
	B=0 # begin state
	#
	# compute the farward and backward
	(lf,lb,ln_eMat,lP)=ln_for_back_mat(hmm,seq,labels) 
	# test if label-path or state-path should be returned
	'''
	if label_list:
		(ap,apl,best_path)=_aposteriori(hmm, lP, lf, lb, hmm.label_list)
		statelabel={}
		for s in E: 
			statelabel[s]=hmm.label_list.index(hmm.states[s].label)
			# create the local apos function 
			apos=lambda i,s,y=apl,st=statelabel: y[i][st[s]] 
	else:
		(ap,apl,best_path)=_aposteriori(hmm, lP, lf, lb)
		# create the local apos function 
		apos=lambda i,s,y=ap: y[i][s]
	'''
	(ln_ap,best_path)=_aposteriori(hmm, lP, lf, lb)
	
	print "log aposteriori",ln_ap
	
	# create the local apos function 
	ln_apos=lambda i,s,y=ln_ap: y[i][s]
	
	###### START PHASE
	ln_f[0][B]=__safe_log(1.0)
	bkt[0][B]=-1
	for s in E:
		ln_f[0][s]=DEF.big_negative
		bkt[0][s]=B
	# from N+B -> N | ARE all equivalent to Begin
	# we are assuming a null can have only a null inlink
	for s in N:
		for t in IN[s]: # IN[s] inludes also B (if this is possible)  
			if ln_f[0][t] > ln_f[0][s]: # there exitsts some inlinks or B
				ln_f[0][s]=ln_f[0][t]
				bkt[0][s]=t
	###### RECURRENCE PHASE for 1 to L-1
	for i in range(1,L):
		# S -> E
		for s in E:
			# update done only if labels are not used
			# or labels corresponds (labels[i-1] is position i !!)
			for t in IS[s]:
				ltmp=ln_f[i-1][t] # ln_a(t,s)+lf[i-1][t]
				if(ln_f[i][s]<ltmp):
					ln_f[i][s]=ltmp
					bkt[i][s]=t
			ln_f[i][s] += ln_apos(i,s) # ap[i][s] a priori  # change aposteriori to return log_probs
		# E -> N . Note the i-level is the same
		for s in N:
			for t in IE[s]:
				ltmp=ln_f[i][t] # ln_a(t,s) +lf[i][t]
				if(ln_f[i][s]<ltmp):
					ln_f[i][s]=ltmp
					bkt[i][s]=t
		# N -> N . Note the i-level is the same
		for s in N:
			for t in IN[s]:
				ltmp=ln_f[i][t] # ln_a(t,s)+lf[i][t]
				if(ln_f[i][s]<ltmp):
					ln_f[i][s]=ltmp
					bkt[i][s]=t
	###### END PHASE 
	
	print "ln_f",ln_f
	print "pointer backtrack matrix",bkt
	
	bestval=DEF.big_negative
	pback=None
	for s in ENDS:
		print "ENDS: %s, lf: %s"%(s,ln_f[L-1][s])
		if(bestval<ln_f[L-1][s]):
			bestval=ln_f[L-1][s]
			pback=s
	if not pback:
		print "ap_viterbi ERRORRRRR!!!"
	###### BACKTRACE
	ln_Probs=[]
	best_path=[]
	i=L-1
	while i >= 0 and pback != B:
		#print  pback, i
		#print apos(i,pback)
		if not label_list:
			ln_Probs.insert(0,ln_apos(i, pback))
		best_path.insert(0,pback)
		ptmp=bkt[i][pback]
		if pback in hmm.emits:
			if label_list:
				ln_Probs.insert(0,ln_apos(i, pback))   
			i-=1
		pback=ptmp
	if returnProbs:
		return(best_path,bestval,ln_Probs)
	else:
		return(best_path,bestval)
		
######################################

def maxAcc_decoder(hmm, seq, returnProbs=None):
	''' decoding algorithm by 
		as described by
		Lukas Kall, Anders Krogh, and Erik L. L. Sonnhammer
		An HMM posterior decoder for sequence feature prediction that includes homology information
		viterbi algorithm on the a posteriori.
		Bioinformatics, Jun 2005; 21: i251 - i257. 
		-> if returnProbs == None
			return(best_state_path,val)
			best_state_path = the best path
			val = the score of the best path
		else
			return(best_state_path,val,returnProbs)
			best_state_path = the best path
			val = the score of the best path
		returnProbs = list of the Probability for each state in best_state_path
		 
	This is not updated for GMMs!!!
	'''
	# settings
	L=len(seq)+1
	lf=NUM.array([[DEF.big_negative]*hmm.num_states]*L,ARRAYFLOAT)
	bkt=NUM.array([[-1]*hmm.num_states]*L,ARRAYINT)
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	S=hmm.topo_order # the list of all state including B
	IS=hmm.in_s # inlinks 
	IE=hmm.in_s_e # inlinks from emittings only
	IN=hmm.in_s_n # inlinks from nulls only
	ENDS=hmm.end_s # end states
	B=0 # begin state
	#
	# compute the farward and backward
	(f,b,eMat,Scale,lP)=for_back_mat(hmm,seq,Scale='Yes',labels=None) 
	# 
	(ap,apl,best_path)=_aposteriori(hmm, lP, f, b, Scale, hmm.label_list)
	statelabel={}
	for s in E: 
		statelabel[s]=hmm.label_list.index(hmm.states[s].label)
	# create the local apos function 
	apos=lambda i,s,y=apl,st=statelabel: y[i][st[s]] 
	###### START PHASE
	lf[0][B]=0.0
	bkt[0][B]=-1
	for s in E:
		lf[0][s]=DEF.big_negative
		bkt[0][s]=B
	# from N+B -> N | ARE all equivalent to Begin
	# we are assuming a null can have only a null inlink
	for s in N:
		for t in IN[s]: # IN[s] inludes also B (if this is possible)  
			if lf[0][t] > lf[0][s]: # there exitsts some inlinks or B
				lf[0][s]=lf[0][t]
				bkt[0][s]=t
				
	###### RECURRENCE PHASE for 1 to L-1
	for i in range(1,L):
		# S -> E
		for s in E:
			# update done only if labels are not used
			# or labels corresponds (labels[i-1] is position i !!)
			for t in IS[s]:
				ltmp=lf[i-1][t] # ln_a(t,s)+lf[i-1][t]
				if(lf[i][s]<ltmp):
					lf[i][s]=ltmp
					bkt[i][s]=t
			lf[i][s]+=apos(i,s) # ap[i][s] a priori
		# E -> N . Note the i-level is the same
		for s in N:
			for t in IE[s]:
				ltmp=lf[i][t] # ln_a(t,s) +lf[i][t]
				if(lf[i][s]<ltmp):
					lf[i][s]=ltmp
					bkt[i][s]=t
		# N -> N . Note the i-level is the same
		for s in N:
			for t in IN[s]:
				ltmp=lf[i][t] # ln_a(t,s)+lf[i][t]
				if(lf[i][s]<ltmp):
					lf[i][s]=ltmp
					bkt[i][s]=t
	###### END PHASE 
	bestval=DEF.big_negative
	pback=None
	for s in ENDS:
		if(bestval<lf[L-1][s]):
			bestval=lf[L-1][s]
			pback=s
	if not pback:
		print "ERRORRRRR!!!"
	###### BACKTRACE
	best_path=[]
	Probs=[]
	i=L-1
	while i >= 0 and pback != B:
		ptmp=bkt[i][pback]
		if pback in hmm.emits:
			Probs.insert(0,apos(i, pback))
			best_path.insert(0,hmm.states[pback].label)
			i-=1
		pback=ptmp
		
	if returnProbs:
		return(best_path,bestval,Probs)
	else:
		return(best_path,bestval)
######################################

def sum_aposteriori(hmm,seq,Scale=None, label_list=[], labels=None):
	''' a posteriori decoding algorithm 	
		sum_aposteriori(hmm,Scale=None, label_list=[]) 
		Scale the scaling vector
		label_list the list of all possible labels
		labels is the sequence labelling used in the forward/backward phases
		-> 
			return(aposteriori_mat,best_path,logProb)
			if label_list != [] then
				best_path contains the best label
				for each sequence position
					and aposteriori_mat the corrisponding probabilities
			else
				best_path contains the name of the best state 
				for each sequence position
					and aposteriori_mat the corrisponding probabilities
					 
		This is not updated for GMMs!!!
	'''
	# settings
	(f,b,eMat,Scale,lP)=for_back_mat(hmm,seq,Scale,labels)
	(apos,aposlabel,bp)=_aposteriori(hmm, lP, f, b, Scale, label_list)
	bestpath=[]
	if(label_list):
		setnames=label_list
		posval=aposlabel[1:] # splice the first unsed postion (Begin)
	else:
		setnames=hmm.state_names
		posval=apos[1:] # splices the first unsed postion (Begin)
	
	for i in bp:
		bestpath.append(setnames[i])

	return(posval,bestpath,lP)

###################################################################

def _aposteriori(hmm, lP, ln_f, ln_b):
	''' a posteriori decoding algorithm 
		_aposteriori(hmm, lP, ln_f, ln_b):
		lP sequence log(probability),
		f forward matrix
		b backward matrix
		NOTE: we consider only the emitting states
		and this function makes explicitly use of the fact that
		the Beginning state is always in the first (0) position
		and it is always SILENT (null)! 
		-> 
			return(log_aposteriori_mat,best_path)
				best_path contains the index of the best state 
				for each sequence position
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	#   since ln_f, ln_b start with dummy position.
	#   This imply that for position i we use seq[i-1] and labels[i-1]
	#   being this two list shorter 
	#    assert len(ln_f) == len(ln_b)
	#P=__safe_log(lP)
	L=len(ln_f)
	start_emit=1 # since th first 0 is B
	first_null=len(hmm.emits)+start_emit # if there are not nulls first_null=num_states (emitting states are topologically ordered before null/silent states)
	best_path=NUM.array([0]*L,ARRAYINT)
	ln_ap=NUM.array([[__safe_log(0.0)]*hmm.num_states]*L,ARRAYFLOAT)
	
	for i in range(L):
		for j in range(hmm.num_states):
			ln_ap[i][j]= ln_f[i][j] + ln_b[i][j] - lP
	
	for i in range(1,L):
		#print "ap_viterbi bestpath compute ap[i],start_emit,first_null,ap[i][start_emit:first_null]",ap[i],start_emit,first_null,ap[i][start_emit:first_null]
		#print "ap_to_list, ap_to_list_slice ",ap_to_list,ap_to_list_slice
		ln_ap_to_list = ln_ap.tolist()[i]
		ln_ap_to_list_slice = ln_ap_to_list[start_emit:first_null]
		best_path[i] = ln_ap_to_list.index(max(ln_ap_to_list_slice))
		#best_path[i]=ap.tolist()[i].index(max(ap[i][start_emit:first_null]))
			 
	# please note the splicing of the B state for the path
	return (ln_ap,best_path[1:])

###################################################################

def one_best_AK(hmm, seq, Scale=None):
	''' one best decoding algorithm which takes advantage of the precalculated emissions eMat
		one_best_AK(hmm, seq, Scale=None) as we understood from Anders Krogh paper
		(Two methods for improving performance of a HMM and their application for gene finding
		ISMB 1997 179-186)
        
		o WARNING: the program may not work properly if there are null states different from
				begin and end        
		-> 
		return(best_label_path,bestval)
			bestval = the value of the best label path
			best_label_path = the best label path (as string og labels)
			
		This is not updated for GMMs!!!
	'''
	# settings
	# PLEASE NOTE THAT seq and labels has different indices
	#   since f, b and Scale start with dummy position.
	#   This imply that for position i we use seq[i-1] and label[i-1]
	#   being this two list shorter 
	eMat=eval_eMatLab(hmm, seq)
	L=len(seq) + 1
	hypSet=[set(),set()]  # the all hypotheses i,i+1
	f=[{},{}] #
	if Scale != None:
		Scale=NUM.array([1.0]*L,ARRAYFLOAT)
#    else:
#        Scale=[]
	end_scale=1.0 # the end scale factor, this is needed for the backward
	a=hmm.a # set the transition probabilities
	E=hmm.emits # the list of the emitting states
	N=hmm.nulls # the list of the silent state 
	S=hmm.topo_order # the list of all state including B
	IS=hmm.in_s # inlinks 
	IE=hmm.in_s_e # inlinks from emittings only
	IN=hmm.in_s_n # inlinks from nulls only
	ENDS=hmm.end_s # end states
	B=0 # begin state
	P=0.0 # P(seq) 
	###### START PHASE
	f[0][(B,() )]=1.0
	for s in E:	
		f[0][(s,() )]=0.0
	# from B -> N 
	for s in N:
		f[0][(s,() )]=a(B,s)
	# from N -> N
	for s in N:	
		for t in IN[s]:
			f[0][(s,() )]+=f[0].get((t,() ),0.0)*a(t,s)
	# if we use the scale factor
	if Scale != None:
		for s in S:
			Scale[0]+=f[0].get((s,() ),0.0)
		for s in S:
			f[0][(s,() )]/=Scale[0]
			
	###### First hypothesis i = 1
	i=1 # 
	for s in E:
		label_s=hmm.states[s].label
		f[1][(s,(label_s,) )]=a(B,s)*eMat[s][0]
	for s in E:
		label_s=hmm.states[s].label
		for t in IS[s]:
			f[1][(s,(label_s,) )]+=a(t,s)*f[0].get((t,() ),0.0)
		f[1][(s,(label_s,) )]*= eMat[s][0]
		hypSet[1].add((label_s,))
    # set transition to null
	# E -> N . Note the i-level is the same
	for s in N:
		for h in hypSet[1]:
			f[1][(s,h)]=0.0
			for t in IE[s]:
				f[1][(s,h)]+=a(t,s)*f[1].get((t,h),0.0)
	# N -> N . Note the i-level is the same
	for s in N:
		for h in hypSet[1]:
			for t in IN[s]:
				f[1][(s,h)]+=a(t,s)*f[1].get((t,h),0.0)
	# if we use the scale factor
	if Scale != None:
		for s in S:
			for h in hypSet[1]:
				Scale[1]+=f[1].get((s,(h) ),0.0)
		for s in S:
			for h in hypSet[1]:
				if Scale[1] > 0 and f[1].has_key((s,(h))): 
					f[1][(s,(h) )]/=Scale[1]
				else:
					f[1][(s,(h) )]=0.0
	###### RECURRENCE PHASE for 2 to L-1
	for i in range(2,L):
		# S -> E
		curr=i%2     # current index
		prev=(i+1)%2 # previous index
		hypSet[curr]=set() #current new hypothesis set
		f[curr]={} # reinitialize
		for s in E:
			label_s=hmm.states[s].label
			max_s=0.0
			hyp=() 
			for h in hypSet[prev]:
				hsum=0.0 # init the hypothesis h
				for t in S:
					hsum+=f[prev].get((t,h),0.0)*a(t,s)
				tuphnew= h+(label_s,)
				f[curr][(s,tuphnew)]=hsum*eMat[s][i-1]
				if hsum > max_s :
					max_s=hsum
					hyp= h+(label_s,)
			hypSet[curr].add(hyp) 
			f[curr][(s,hyp)]= max_s*eMat[s][i-1] # remember seq[i-1] is x_i        
		# E -> N . Note the i-level is the same	
		# N -> N . Note the i-level is the same
		for s in N:
			for h in hypSet[curr]:
				f[curr][(s,h)]=0.0
				for t in IE[s]+IN[s]:
					f[curr][(s,h)]+=a(t,s)*f[curr].get((t,h),0.0)
		# if we use the scale factor
		if Scale != None:
			Scale[i]=0.0
			cVal=0.0
			for h in hypSet[curr]:
				for s in S:
					cVal+=f[curr].get((s,h ),0.0)
				if cVal > Scale[i]:
					Scale[i]=cVal
			for s in S:
				for h in hypSet[curr]:
					if Scale[i]>0 and f[curr].has_key((s,h)):
						f[curr][(s,h )]/=Scale[i]
					else:
						f[curr][(s,h )]=0.0
#        print "#",i,len(hyp)	
	###### END PHASE 
	max_hyp=0.0
	hyp=() 
	for h in hypSet[curr]:
		v_hyp=0.0
		for s in hmm.end_s:  # emitting end states
			v_hyp+=f[curr].get((s,h),0.0)
		if v_hyp > max_hyp:
			max_hyp=v_hyp
			hyp=h
	return hyp,max_hyp
# activate if you need probability
#    if(Scale != None):
#        lP=__safe_log(val)+ NUM.sum(NUM.log(Scale))
#    else:
#        lP=__safe_log(val)
#    
######################################

###################################################################
######################################
#
# LEARNING
#
######################################

def __expected_mat_transitions(hmm,AC,S,OE,ON,o,ln_a,Scale):
	'''
	__expected_mat_transitions(hmm,AC,S,OE,ON,o,a,e,Scale):
		o hmm -> the hmm to train
		o AC -> matrix with the number of expected transitions
		o S -> state list
		o OE -> state emitting outlinks
		o ON -> state silent outlinks
		o o  -> trainable object
		o a -> a(i,j) transition prob. function
		o Scale -> scale vector or None
	'''
	for s in S: 
		# transitions
		s_trpos=hmm.effective_tr_pos[s]
		for t in OE[s]: # emitting states
			t_trpos=hmm.effective_tr_pos[t]
			ln_sum_curr_prot=__safe_log(0.0)
			for i in range(0,o.len): # from position 1 of seq to the last
				ln_prod = o.ln_f[i][s] + ln_a(s,t) + o.ln_eMat[t][i] + o.ln_b[i+1][t]
				ln_sum_curr_prot = __e_ln_sum(ln_sum_curr_prot, ln_prod)	
			AC[s_trpos][t_trpos] += __safe_exp(ln_sum_curr_prot - o.lprob)
		for t in ON[s]: # null states
			t_trpos=hmm.effective_tr_pos[t]
			ln_sum_curr_prot=__safe_log(0.0)
			for i in range(0,o.len+1): # from position 1 of seq to the last
				ln_prod = o.ln_f[i][s] + ln_a(s,t) + o.ln_b[i][t]
				ln_sum_curr_prot = __e_ln_sum(ln_sum_curr_prot, ln_prod)	
			AC[s_trpos][t_trpos] += __safe_exp(ln_sum_curr_prot - o.lprob)
	
	return AC
	
##############################################

def __disc_set_param(hmm,ACc,ECc,ACf,ECf,epsilon=1.0):
	'''
	__disc_set_param(hmm,ACc,ECc,ACf,ECf,epsilon=0.01):
		o hmm -> Hidden Markov Model to set
		o ACc   -> matrix with the number of expected transitions (clamped phase)
		o ECc   -> matrix with the number of expected emissions (clamped phase)
		o ACf   -> matrix with the number of expected transitions (free phase)
		o ECf   -> matrix with the number of expected emissions (free phase)
	'''   
	# find D
	D=0.0
	delta=0.0
	for s in hmm.topo_order: # here we do not take advantage of tr_updatable
		s_trpos=hmm.effective_tr_pos[s]
		if not hmm.fix_tr[s]: # we can update the transitions
			for t in hmm.out_s[s]:
				t_trpos=hmm.effective_tr_pos[t]
				dc=ACc[s_trpos][t_trpos]-ACf[s_trpos][t_trpos]
				if hmm.a(s,t) > 0:
					delta=abs(dc)/hmm.a(s,t)
				else:
					print >> sys.stderr, "zero transition for",hmm.state_names[s],hmm.state_names[t]
				if delta > D:
					D=delta
	for s in hmm.em_updatable:
		emlet=hmm.states[s].em_letters
		empos=hmm.effective_em_pos[s]
		for c in range(len(emlet)):
			xc=emlet[c]
			dc=ECc[empos][c]-ECf[empos][c]
			if hmm.e(s,xc) > 0:
				delta=abs(dc)/hmm.e(s,xc)
			else:
				print >> sys.stderr, "zero emission for",hmm.state_names[s],xc,hmm.e(s,xc)
			if delta > D:
				D=delta
				
	# transitions
	D+=epsilon
	diffParam=0.0
	nparm=0
	for s in hmm.topo_order: # here we do not take advantage of tr_updatable
		s_trpos=hmm.effective_tr_pos[s]
		if not hmm.fix_tr[s]: # we can update the transitions
			Esc_Esf=0.0
			for t in hmm.out_s[s]:
				t_trpos=hmm.effective_tr_pos[t]
				dc=ACc[s_trpos][t_trpos]-ACf[s_trpos][t_trpos]
				Esc_Esf+=dc
			for t in hmm.out_s[s]:
				t_trpos=hmm.effective_tr_pos[t]
				tmp=(ACc[s_trpos][t_trpos]-ACf[s_trpos][t_trpos]+hmm.a(s,t)*D)
				if tmp>0 and Esc_Esf+D>0 :
					newpar=tmp/(Esc_Esf+D)	
					dpar=newpar-hmm.a(s,t)
					diffParam+=dpar*dpar
					hmm.set_a(s,t,newpar)
					nparm+=1
				else:
					print >> sys.stderr, "tmp<=0 or Esc_Esf+D<=0",hmm.state_names[s],hmm.state_names[t],tmp,Esc_Esf+D
		# else:
		# sys.stderr.write("TR "+hmm.states[s].name+ " "+hmm.states[t].name+" zerooo\n")
	# emission 
	for s in hmm.em_updatable:
		emlet=hmm.states[s].em_letters
		empos=hmm.effective_em_pos[s]
		Esc_Esf=0.0
		for c in range(len(emlet)):
			xc=emlet[c]
			dc=ECc[empos][c]-ECf[empos][c]
			Esc_Esf+=dc
		for c in range(len(emlet)):
			xc=emlet[c]
			tmp=ECc[empos][c]-ECf[empos][c]+hmm.e(s,xc)*D
			if tmp>0 and Esc_Esf+D>0:
				newpar=tmp/(Esc_Esf+D)
				dpar=newpar-hmm.e(s,xc)
				diffParam+=dpar*dpar
				hmm.set_e(s,xc,newpar)
				nparm+=1
			else:
				print >> sys.stderr, "tmp<=0 or Esc_Esf+D<=0",hmm.state_names[s],xc,tmp,Esc_Esf+D
	return NUM.sqrt(diffParam/nparm)
#            else:
#                sys.stderr.write("EM "+hmm.states[s].name+" "+emlet[c]+" zerooo\n")

##############################################
def __disc_Riis_set_param(hmm,ACc,ECc,ACf,ECf,epsilon=0.001):
	'''
	__disc_Riis_set_param(hmm,ACc,ECc,ACf,ECf,epsilon=0.001):
		o hmm -> Hidden Markov Model to set
		o ACc   -> matrix with the number of expected transitions (clamped phase)
		o ECc   -> matrix with the number of expected emissions (clamped phase)
		o ACf   -> matrix with the number of expected transitions (free phase)
		o ECf   -> matrix with the number of expected emissions (free phase)
	'''   
	# transitions 
	diffParam=0.0
	nparm=0
	for s in hmm.topo_order: # here we do not take advantage of tr_updatable
		s_trpos=hmm.effective_tr_pos[s]
		if not hmm.fix_tr[s]: # we can update the transitions
			Esc_Esf=0.0
			xv=NUM.array([0.0]*len(hmm.out_s[s]))
			iv=0
			for t in hmm.out_s[s]:
				t_trpos=hmm.effective_tr_pos[t]
				xv[iv]=ACc[s_trpos][t_trpos]-ACf[s_trpos][t_trpos]
				Esc_Esf+=xv[iv]
				iv+=1
			iv=0
			normfact=0.0
			for t in hmm.out_s[s]:         
				xv[iv]-=hmm.a(s,t)*Esc_Esf          # compute x= Aij^C-Aij -a(s,t)*[Ei^c - Ei]
				xv[iv]=hmm.a(s,t) * NUM.exp(epsilon*xv[iv])  # compute x= hmm.a(s,t) * exp(epsilon*x)
				normfact+=xv[iv]
				iv+=1
			iv=0
			for t in hmm.out_s[s]:
				newpar=xv[iv]/normfact
				dpar=newpar-hmm.a(s,t)
				diffParam+=dpar*dpar
				hmm.set_a(s,t,newpar)
				iv+=1
				nparm+=1
	# emission 
	for s in hmm.em_updatable:
		emlet=hmm.states[s].em_letters
		empos=hmm.effective_em_pos[s]
		Esc_Esf=0.0
		xv=NUM.array([0.0]*len(emlet))
		iv=0
		for c in range(len(emlet)):
			xc=emlet[c]
			xv[iv]=ECc[empos][c]-ECf[empos][c]
			Esc_Esf+=xv[iv]
			iv+=1
		iv=0
		normfact=0.0
		for c in range(len(emlet)):
			xc=emlet[c]
			xv[iv]-=hmm.e(s,xc)*Esc_Esf # compute x= Eic^C-Eic -e(s,c)*[Ei^c - Ei]
			xv[iv]=hmm.e(s,xc)*NUM.exp(epsilon*xv[iv]) # compute x=e(s,c)*exp(epsilon*x)
			normfact+=xv[iv]
			iv+=1
		iv=0
		for c in range(len(emlet)):
			xc=emlet[c]
			newpar=xv[iv]/normfact
			dpar=newpar-hmm.e(s,xc)
			diffParam+=dpar*dpar
			hmm.set_e(s,xc,newpar)
			iv+=1
			nparm+=1
	return NUM.sqrt(diffParam/nparm)
#            else:
#                sys.stderr.write("EM "+hmm.states[s].name+" "+emlet[c]+" zerooo\n")

#####################################################

def __set_param(hmm,AC,EC):
	'''
	__set_param(S,OS,hmm,AC,EC)
		o hmm -> Hiddn Markov Model to set
		o AC   -> matrix with the number of expected transitions
		o EC   -> matrix with the number of expected emissions
	'''   
	# normalize and set
	for s in hmm.topo_order: # here we do not take advantage of tr_updatable
		s_trpos=hmm.effective_tr_pos[s]
		# transitions
		if not hmm.fix_tr[s]: # we can update the transitions
			tmp=0.0
			for t in hmm.out_s[s]:
				t_trpos=hmm.effective_tr_pos[t]
				tmp+=AC[s_trpos][t_trpos]
			for t in hmm.out_s[s]:
				t_trpos=hmm.effective_tr_pos[t]
				if tmp:
					hmm.set_a(s,t,AC[s_trpos][t_trpos]/tmp)
				# else:
					# sys.stderr.write("TR "+hmm.states[s].name+ " "+hmm.states[t].name+" zerooo\n")
		# emission 
	for s in hmm.em_updatable:
		emlet=hmm.states[s].em_letters
		empos=hmm.effective_em_pos[s]
		tmp=NUM.sum(EC[empos])  # for each updatable state the expected emission values are summed over all observation sequence positions
		for c in range(len(emlet)):
			if tmp:
				hmm.set_e(s,emlet[c],EC[empos][c]/tmp)
			# else:
				# sys.stderr.write("EM "+hmm.states[s].name+" "+emlet[c]+" zerooo\n")
				
#####################################################

def __gmm_set_param(hmm,set_of_trobj,AC,WC,MC,UC):
    '''
        __set_param_gmm(hmm,AC,WC,MC,UC)
            o hmm -> Hiddn Markov Model to set
            o AC   -> matrix with the number of expected transitions
            o WC   -> matrix with the number of expected weights
            o WC= expected number for weights for the mixtures
            o MC= expected number for means for the mixtures
            o UC= expected number for covars for the mixtures 		   
    '''
    dimProfile = hmm.dimProfile #dimension of the observation vector profiles
    # normalize and set
    for s in hmm.topo_order: # here we do not take advantage of tr_updatable
        s_trpos=hmm.effective_tr_pos[s]
        # transitions
        if not hmm.fix_tr[s]: # we can update the transitions
            tmp=0.0
            for t in hmm.out_s[s]:
                t_trpos=hmm.effective_tr_pos[t]
                tmp+=AC[s_trpos][t_trpos]
            for t in hmm.out_s[s]:
                t_trpos=hmm.effective_tr_pos[t]
                if tmp:
                    hmm.set_a(s,t,AC[s_trpos][t_trpos]/tmp)
                #else:
                    #sys.stderr.write("TR "+hmm.states[s].name+ " "+hmm.states[t].name+" zerooo\n")
    
    # emission 
    for s in hmm.em_updatable:
        empos=hmm.effective_em_pos[s]
        curr_state_em_obj = hmm.states[s].get_emissions()  # returns a node_em_gmm object as we are dealing with GMMs
        curr_state_mix_num = curr_state_em_obj.get_mix_num()
        for k in range(curr_state_mix_num): #update each mixture in current state s
            hmm.set_mix_weight(s,k,WC[empos][k]/NUM.sum(WC[empos])) # set the new k-th mixture weight after normalising
            # check if the current state's mixture densities are updatable (either fixed or tied with another state)
            # we use the same state name for the emissions at state level and at mixture level due to the method of 
            # tying employed
            if s in hmm.em_mix_updatable:
            	em_mix_pos = hmm.effective_em_mix_pos[s]
            	#find all the states with tied mixture densities and sum the denominator terms
            	#found in the update formula (see Digalakis' paper)
            	inv_eff_em_mix_pos = {}
            	for key, v in hmm.effective_em_mix_pos.iteritems():
            		inv_eff_em_mix_pos[v] = inv_eff_em_mix_pos.get(v, [])
            		inv_eff_em_mix_pos[v].append(key)
            	
            	#mix_norm_ind = NUM.where(NUM.array(hmm.effective_em_mix_pos) == em_mix_pos)[0].tolist()
            	mix_norm_ind = inv_eff_em_mix_pos[em_mix_pos]
            	norm_factor = 0.0
            	for state_id in mix_norm_ind:
            		norm_empos=hmm.effective_em_pos[state_id] #this must be the effective state pos, not the effective mixture pos
            		norm_factor += WC[norm_empos][k]
            		
            	curr_mean = MC[em_mix_pos][k]/norm_factor
            	#the empirical covariance is updated using the newly updated mean parameters
            	#iterate through all of the equivalent states (states with tied mixtures)
            	for state_id in mix_norm_ind:
            		for j in range(len(set_of_trobj)):
            			o = set_of_trobj[j]
            			sum_curr_prot_c = NUM.zeros([dimProfile,dimProfile],ARRAYFLOAT) # covars
            			for i in range(1,o.len+1):
            				curr_prot_c_inc = NUM.zeros([dimProfile,dimProfile],ARRAYFLOAT) # covars
            				#update the covar for the current mixture and observation position
            				tmp_arr = (NUM.asarray(o.seq[i-1]) - NUM.asarray(curr_mean))
            				#we use numpy's broadcasting abilities to perform the array*array.T operation for the covars
            				tmp_covar = tmp_arr[:,NUM.newaxis]*tmp_arr
            				curr_prot_c_inc = __safe_exp(o.ln_gamma.get((i-1,state_id,k),__safe_log(0.0)))*tmp_covar
            				#update the covariance for whole observation sequence
            				sum_curr_prot_c += curr_prot_c_inc
            			if(o.scale == None):
            				UC[em_mix_pos][k] += __safe_exp(__safe_log(sum_curr_prot_c) - o.lprob)
            			else:
            				UC[em_mix_pos][k] += sum_curr_prot_c
            	#final empirical covariance for all sequences and positions in each sequence for this state and mixture        
            	emp_cov = UC[em_mix_pos][k]/norm_factor
            	#compute and set the covariance precision matrices corresponding to the newly updated empirical covariance matrices
            	# here we use the empirical covariance as input to the graphlasso algorithm, with a predefined
            	# alpha: positive float
            	# The regularization parameter: the higher alpha, the more regularization, the sparser the inverse covariance
            	glasso_cov, glasso_precision = graph_lasso(emp_cov, DEF.alpha, mode='cd', tol=1e-4, max_iter=100,verbose=False)
            	hmm.set_mix_density(s,k,curr_mean,glasso_cov,glasso_precision) # set the new k-th mixture density function after normalisin
        
        already_normalised = hmm.normalise_mix_weights(s) #ensure that the current states mixture weights sum to one
        if not already_normalised:
        	print "State %d did not have its mixture weights normalised"%s, hmm.states[empos].name
    print "GMM parameters updated."
    
#####################################################

def init_AC_EC(hmm,pseudocount):
	'''
	init_AC_EC(hmm,pseudocount)
		o hmm -> Hiddn Markov Model to set
		o pseudocount -> pseudocount to add
		=> return AC,EC
			where AC= expected number of transitions 
			EC= expected number of emissions 
	'''
	dim_em_alphabet=len(hmm.emission_alphabet)
	num_tr=len(hmm.effective_tr) # number of effective transitions (related to node_tr)
	num_em=len(hmm.effective_em) # number of effective emissions (related to node_em) 

	AC=NUM.array([[pseudocount]*num_tr]*num_tr,ARRAYFLOAT)
	EC=NUM.array([[pseudocount]*dim_em_alphabet]*num_em,ARRAYFLOAT)
	return AC,EC

def init_AC_WC_MC_UC(hmm,pseudoAC,pseudoWC,pseudoMC,pseudoUC,set_of_trobj=[],use_clustering=False):
    '''
        init_AC_WC_MC_UC(hmm,pseudoAC,pseudoWC,pseudoMC,pseudoUC,set_of_trobj=[],use_clustering=False)
            o hmm -> Hiddn Markov Model to set
            o pseudocount -> random pseudocount to initialise model parameters 
            o set_of_trobj = the list of trobj protein sequence vector profiles used for training
                these are used for kmeans segmentation clustering
            o use_clustering = boolean indicator for random or kmeans/spectral initilisation
            => return AC,WC,MC,UC
                where AC= expected number for transitions 
                    WC= expected number for weights for the mixtures
                    MC= expected number for means for the mixtures
                    UC= expected number for covars for the mixtures 
        The AC, WC, MC, UC can be lists of vectors/matrices for initialisation purposes as derived by 
        Rabiner's k-means segmentation
    '''
    num_tr=len(hmm.effective_tr) # number of effective transitions (related to num node_tr)
    E=hmm.emits # the list of all emitting states
    dimProfile = hmm.dimProfile #dimension of the observation vector profiles	
    AC=NUM.array([[pseudoAC]*num_tr]*num_tr,ARRAYFLOAT)
    #GMM emission parameters (key,param) where key is the name (effective_em_pos) of one of the effect emissions (num of node_em)
    # and param is either a mixture weight, mixture mean vector, or mixture covar matrix
    WC = {}
    MC = {}   # make sure that the components are floats
    UC = {}
    
    if not use_clustering:   #set each component of each parameter to have the corresponding pseudo-count value, covars are set as diagonal matrices	
        for s in E:
            # emissions
            empos=hmm.effective_em_pos[s]
            curr_state_em_obj = hmm.states[s].get_emissions()
            curr_state_mix_num = curr_state_em_obj.get_mix_num()
            WC[empos] = NUM.array([pseudoWC]*curr_state_mix_num)
            em_mix_pos = hmm.effective_em_mix_pos[s]
            MC[em_mix_pos] = NUM.array([[pseudoMC]*dimProfile]*curr_state_mix_num)
            UC[em_mix_pos] = NUM.array([NUM.diag([pseudoUC]*dimProfile)]*curr_state_mix_num)
    else:
        WC,MC,UC = _clustering(hmm,set_of_trobj,pseudoWC,pseudoMC,pseudoUC)  #perform Rabiner's kmeans segmentation clustering
        
    return AC,WC,MC,UC     
	
#####################################################
		
def _clustering(hmm,set_of_trobj,pseudoWC,pseudoMC,pseudoUC,clust_init=5):
    '''
        Performs the kmeans/spectral clustering as described by Rabiner.
        This uses the best state path as obtained from the Viterbi algorithm
        computed using the current Hidden Markov Model. The hidden state path is
        used to partition observation profile vectors (corresponding to a position in a protein 
        sequence) into sets. These sets are then clustered using the kmeans/sepectral algorithm 
        corresponding to the number of Gaussian Mixtures in the respective hidden states.
        
        returns WC, MC, UC  for each hidden state and mixture in the corresponding states
    '''

    E=hmm.emits # the list of all emitting states
    dimProfile = hmm.dimProfile #dimension of the observation vector profiles	    
    state_col = {} #dictionary of (key,value) = (states, arrays of observation vector profiles)
    min_num_vecs = 2 #mininum number of vectors needed for sklearn.kmeans algorithm to determine various clusters
        
    #initial parameter output variables
    WC = {} 
    MC = {}
    UC = {}
    
    for s in E:
    	state_col[s] = NUM.array([])
    
    for cur_trobj in set_of_trobj:  # for the current protein sequence obtain the viterbi path 
        cur_seq = cur_trobj.seq 
        #best_path,bestval,logProbPath = _viterbi(hmm,cur_seq) #best_path is a list of hidden states, states are represented by integers corresponding to the model grammar file
        best_path,bestval = ap_viterbi(hmm,cur_seq) #best_path is a list of hidden states, states are represented by integers corresponding to the model grammar file
        #best_path,bestval = viterbi(hmm,cur_seq) #best_path is a list of hidden states, states are represented by integers corresponding to the model grammar file
        '''
        j = 0
        for i in range(len(best_path)):
            cur_state = best_path[i]
            if cur_state in E: #must be an emitting state, otherwise it is a silent/null state so skip
                cur_arr = state_col.get(cur_state,NUM.array([])) # get the array of observation profiles otherwise create new empty array
                if cur_arr.shape[0] == 0:
                    cur_arr = NUM.array([cur_seq[j]]) # first observation vector profile to be added to this state's collection
                else:
                    cur_arr = NUM.append(cur_arr,NUM.array([cur_seq[j]]),axis=0)
                state_col[cur_state] = cur_arr # reset the states collection of observation vector profiles
                j += 1 #only count the sequence observation profiles which are emitted
            #else:
            #    print "silent state at pos %d \n"%i
        '''
    	for i in range(len(cur_seq)):
    		cur_state = best_path[i]
    		if cur_state in E: #must be an emitting state, otherwise it is a silent/null state so skip
    			cur_arr = state_col.get(cur_state,NUM.array([])) # get the array of observation profiles otherwise create new empty array
    			if cur_arr.shape[0] == 0:
    				cur_arr = NUM.array([cur_seq[i]]) # first observation vector profile to be added to this state's collection
    			else:
    				cur_arr = NUM.append(cur_arr,NUM.array([cur_seq[i]]),axis=0)
    			state_col[cur_state] = cur_arr # reset the states collection of observation vector profiles
            #else:
            #    print "silent state at pos %d \n"%i
    	
    #scaling for tied states and mixtures
    empos_scale = {}
    em_mix_pos_scale = {}
    #perform the kmeans clustering using sklearn.cluster.KMeans implementation (possibly try spectral clustering if number of elements in the mixtures is small or few mixtures)
    for cur_state,cur_arr in state_col.items():
        cur_mix_num = hmm.states[cur_state].get_emissions().get_mix_num()
        num_vecs = cur_arr.shape[0] #number of vectors predicted in state j (cur_state)
        
        #effective state position, as used by other HMM functions
        empos=hmm.effective_em_pos[cur_state]
        em_mix_pos = hmm.effective_em_mix_pos[cur_state]
        
        WC[empos] = WC.get(empos,NUM.array([0.0]*cur_mix_num))
        MC[em_mix_pos] = MC.get(em_mix_pos,NUM.array([[0.0]*dimProfile]*cur_mix_num))
        UC[em_mix_pos] = UC.get(em_mix_pos,NUM.array([NUM.diag([1.0]*dimProfile)]*cur_mix_num))
        	
        #set scaling factors for the tied states and mixtures
        if (WC[empos] == NUM.array([0.0]*cur_mix_num)).all():
        	empos_scale[empos] = 1.0 #define
        else:
        	empos_scale[empos] += 1.0 #update for number of tied states
        
        if (MC[em_mix_pos] == NUM.array([[0.0]*dimProfile]*cur_mix_num)).all():
        	em_mix_pos_scale[em_mix_pos] = 1.0 #define
        else:
        	em_mix_pos_scale[em_mix_pos] += 1.0 #update for number of tied mixtures
        
        # check the min number of observation profile vectors have been assigned to the current state
        if num_vecs >= min_num_vecs:
        	#kmeans clustering
        	#This can lead to numerical instability for the Baum-Welch EM algorithm as the various state clusters might be sparse
        	kmeans_est = sk_kmeans(init='k-means++', n_clusters=cur_mix_num, n_init=clust_init)
        	pred_mix_indices = kmeans_est.fit_predict(cur_arr) #fit and predict the mxiture labels for the observation profile vectors
        	#spectral clustering
        	#This option needs a large amount of memory to function efficiently otherwise paging takes place
        	#spectral_est = sk_spectralclust(n_clusters=cur_mix_num, n_init=clust_init)
        	#pred_mix_indices = spectral_est.fit_predict(cur_arr)
			
        	for k in range(cur_mix_num):
        		ans_ind = NUM.where(pred_mix_indices == k) #obtain observation profiles predicted to be in mixture k
        		vecs_j_m = cur_arr[ans_ind]
        		c_j_m = vecs_j_m.shape[0]/float(num_vecs)
        		mean_j_m = NUM.mean(vecs_j_m,axis=0)
        		#cov_j_m = NUM.cov(vecs_j_m,rowvar=0)		#compute estimates for the emission parameters 
        		# must use sparse inverse covariance estimation by graphical lasso method as the determinant of a 
        		# non-diagonal covariance matrix will be "too small" and cause numerical issues
        		model = GraphLassoCV(cv=1)
        		model.fit(vecs_j_m)
        		cov_j_m = model.covariance_
        		WC[empos][k] += c_j_m
        		MC[em_mix_pos][k] += mean_j_m
        		
        		#check if the matrix is symmetric and positive definite as numpy.cov() only ensures that the variance-covariance matrix is symmetric semi-postive definite
        		update_cov = True
        		try:
        			L = NUM.linalg.cholesky(cov_j_m)
        		except:
        			pd_cov_j_m = nearPD_Higham.nearPD(cov_j_m) # uses Higham's algorithm to compute the nearest positive definite matrix
        			UC[em_mix_pos][k] += pd_cov_j_m # replace not positive semi-definite covar with nearest matrix that is positive semi-definite
        			update_cov = False
        		if update_cov:
        			UC[em_mix_pos][k] += cov_j_m
        else:
        	#if there are insufficient observation profile vectors assigned to this state then
        	#use random initialisation with pseudo counts as defined by the user for the initial cycle/epoch
        	WC[empos] += NUM.array([pseudoWC+0.1]*cur_mix_num)
        	MC[em_mix_pos] += NUM.array([[pseudoMC+0.1]*dimProfile]*cur_mix_num)
        	UC[em_mix_pos] += NUM.array([NUM.diag([pseudoUC+0.1]*dimProfile)]*cur_mix_num)
        	print "State %d has zero or one observation profile vector assigned to it via the A-Priori Viterbi algorithm. This might indicate that the model grammar does not correctly represent the experimental design."%(cur_state)
    
    #scaling for tied states and mixtures
    #need to iterate through all dictionary items and scale the numpy.array elements individually
    '''
    WC = {key:val/float(empos_scale[key]) for key, val in WC.items()}
    MC = {key:val/float(em_mix_pos_scale[key]) for key, val in MC.items()}
    UC = {key:val/float(em_mix_pos_scale[key]) for key, val in UC.items()}    
    '''
    #dictionary comprehensions are not supported in py2.6.6 only from py2.7 up
    for key,val in WC.items():
    	WC[key] = val/float(empos_scale[key])
    for key, val in MC.items():
    	MC[key] = val/float(em_mix_pos_scale[key])
    for key, val in UC.items():
    	UC[key] = val/float(em_mix_pos_scale[key])  
    
    #return kmeans segmentation clustering estimates
    return WC,MC,UC	
		
#####################################################

def _update_Vit_AC_EC(AC,EC,hmm,seq,bp,pseudocount):
	''' 
	update_Vit_AC_EC(AC,EC,hmm,best_path,pseudocount)
		o AC ->  expected number of transitions 
		o EC -> expected number of emissions
		o hmm -> Hiddn Markov Model to set
		o seq -> sequence to model
		o bp  -> Viterbi best path
		o pseudocount -> pseudocount to add default = 0.0
		=> set the new hmm transition and emission probabilities
		using the Viterbi Learning
	'''
	# init transitions and emissions counts 
	for i in range(1,len(bp)):  # the first is begin and does not emit
		p_pos=hmm.effective_tr_pos[bp[i-1]]
		c_pos=hmm.effective_tr_pos[bp[i]]
		AC[p_pos][c_pos]+=1 #update transition
		if bp[i] in hmm.emits:
			empos=hmm.effective_em_pos[bp[i]]
			if(type(seq[0]) != type('String')): 
				k=0
				for c in hmm.emission_alphabet:
					EC[empos][k]+=seq[i-1][k]*hmm.e(c_pos,c)
					k+=1
			else:
				c=hmm.emission_alphabet.index(seq[i-1])
				EC[empos][c]+=1
				
	return  

#===============
def __symbol_update_AC_EC(AC,EC,hmm,o,Scale,pseudocount):
	''' 
	__symbol_update_AC_EC(AC,EC,hmm,o,Scale,pseudocount)
		o AC ->  expected number of transitions 
		o EC -> expected number of emissions
		o hmm -> Hiddn Markov Model to set
		o o -> trainable object
		o Scale -> scale vector or None 
		o pseudocount -> pseudocount to add default = 0.0
		=> set the new hmm transition and emission probabilities
		using the classical way (Durbin et al. 1998)
	'''
	# init transitions and emissions counts 
	E=hmm.emits # the list of all emitting states
	S=hmm.topo_order # the list of all state including B
	OE=hmm.out_s_e # emitting outlinks
	ON=hmm.out_s_n # silent outlinks
	OS=hmm.out_s # outlinks
	
	# transitions
	# Please note that we are invoking using the      hmm methods
	#                                                  \      \
	__expected_mat_transitions(hmm,AC,S,OE,ON,o,hmm.a,Scale)
	for s in E:
		# emissions
		empos=hmm.effective_em_pos[s]
		for c in hmm.states[s].em_letters:
			cpos=hmm.emission_alphabet.index(c)
			sum_curr_prot=0.0
			for i in range(1,o.len+1): # from position 2 of seq to the last
				if(o.seq[i-1] == c):
					if(Scale != None):
						sum_curr_prot+=o.f[i][s]*o.b[i][s]*o.scale[i]
					else:
						sum_curr_prot+=o.f[i][s]*o.b[i][s]
			if(Scale == None):
				o.prob=NUM.exp(o.lprob)
				EC[empos][cpos]+=sum_curr_prot/o.prob
			else:
				EC[empos][cpos]+=sum_curr_prot
				
###############################

def __vec_update_AC_EC(AC,EC,hmm,o,Scale,pseudocount):
	''' 
	__vec_update_AC_EC(AC,EC,hmm,o,Scale,pseudocount)
		o AC ->  expected number of transitions 
		o EC -> expected number of emissions
		o hmm -> Hiddn Markov Model to set
		o o -> trainable object
		o Scale -> scale vector or None 
		o pseudocount -> pseudocount to add default = 0.0
		=> set the new hmm transition and emission probabilities
		using the classical way (Durbin et al. 1998)
	'''
	S=hmm.topo_order # the list of all state including B
	E=hmm.emits # the list of all emitting states
	OS=hmm.out_s # outlinks
	OE=hmm.out_s_e # emitting outlinks
	ON=hmm.out_s_n # silent outlinks
    
	# transitions
	# Please note that we are invoking using the      hmm methods
	#                                                     \    \ 
	__expected_mat_transitions(hmm,AC,S,OE,ON,o,hmm.a,Scale)
	if(Scale != None):
		fT=NUM.transpose(o.f) # fT = transpose of forward matrix
		bT=NUM.transpose(o.b) # bT = transpose of backward matrix
		seqT=NUM.transpose(o.seq) # seqT = transpose of the sequnce vector
		for s in E:
			# emissions
			# we count all the cases Gigi 2002
			# now we found the index of the current emission
			empos=hmm.effective_em_pos[s]
			# matrix multiplication
			EC[empos]+=NUM.dot(seqT,fT[s][1:]*bT[s][1:]*o.scale[1:])
#           This is equivalent to
#             for c in range(dim_em_alphabet):
#                for i in range(1,o.len+1): # from position 1 of seq to the last
#                    sum_curr_prot+=o.f[i][s]*o.b[i][s]*o.scale[i]*o.seq[i-1][c]
#                    EC[empos][c]+=sum_curr_prot
	else: 
		fT=NUM.transpose(o.f) # fT = transpose of forward matrix
		bT=NUM.transpose(o.b) # bT = transpose of backward matrix
		seqT=NUM.transpose(o.seq) # seqT = transpose of the sequnce vector
		o.prob=NUM.exp(o.lprob)
		for s in E:
			# emissions
			# we count all the cases Gigi 2002
			# now we found the index of the current emission
			empos=hmm.effective_em_pos[s]
			EC[empos]+=(NUM.dot(seqT,fT[s][1:]*bT[s][1:])/o.prob)
#           This is equivalent to
#                for c in range(dim_em_alphabet):
#                   for i in range(1,o.len+1): # from position 1 of seq to the last
#                       sum_curr_prot+=o.f[i][s]*o.b[i][s]*o.seq[i-1][c]
#                    EC[empos][c]+=sum_curr_prot/o.prob
	# normalize and set
	
###############################
def __gmm_update(AC,WC,MC,UC,hmm,o,Scale):
    '''
    __gmm_update(AC,WC,MC,UC,hmm,o,Scale)
    o AC= expected number for transitions 
    o WC= expected number for weights for the mixtures
    o MC= expected number for means for the mixtures
    o UC= expected number for covars for the mixtures 
    o hmm -> Hidden Markov Model to set
    o o -> trainable object
    o Scale -> scale vector or None 
    o pseudocount -> don't really use pseudocount here
    => set the new hmm transition and gmm emission parameters
    using the classical way (Rabiner)
    '''
    S=hmm.topo_order # the list of all state including B
    E=hmm.emits # the list of all emitting states
    OS=hmm.out_s # outlinks
    OE=hmm.out_s_e # emitting outlinks
    ON=hmm.out_s_n # silent outlinks
    dimProfile = hmm.dimProfile #dimension of the observation vector profiles
    o.ln_gamma = {} #previous gamma values must be cleared, these newly updated values will be used to compute the new covariance matrices
    # transitions
    # Please note that we are invoking using the      hmm methods
    #                                                     \    \ 
    __expected_mat_transitions(hmm,AC,S,OE,ON,o,hmm.ln_a,Scale)
    for s in E:
        # emissions
        empos=hmm.effective_em_pos[s]
        curr_state_em_obj = hmm.states[s].get_emissions()  # returns a node_em_gmm object as we are dealing with GMMs
        curr_state_mix_num = curr_state_em_obj.get_mix_num()
        em_mix_pos = hmm.effective_em_mix_pos[s]
        for k in range(curr_state_mix_num): #update each mixture in current state s
            curr_weight = curr_state_em_obj.get_mix_weight(k) #float value
            curr_mix_density = curr_state_em_obj.get_mixture_density(k) #mixture_density object
            ln_sum_curr_prot_w = __safe_log(0.0)  # weights
            sum_curr_prot_m = NUM.array([0.0]*dimProfile,ARRAYFLOAT) # means
            for i in range(1,o.len+1): # from position 2 of seq to the last (for each position in the sequence)
                #update the weight for the current mixture
                ln_curr_prot_w_inc = __safe_log(0.0)
                curr_prot_m_inc = NUM.array([0.0]*dimProfile,ARRAYFLOAT) # means
                '''
                if(Scale != None):
                    curr_prot_w_inc = (o.f[i][s]*o.b[i][s]*o.scale[i])*curr_weight*curr_mix_density.em(o.seq[i-1])
                else:
                    curr_prot_w_inc = (o.f[i][s]*o.b[i][s])*curr_weight*curr_mix_density.em(o.seq[i-1])
                '''
            	#normalise for the i-th sequence position over all mixtures in this state, use precomputed emission matrix eMat
                #sequence position values are (i-1)th position!
                '''
                curr_prot_w_inc /= o.eMat[s][i-1] # same as using hmm.e(s,o.seq[i-1])          
                '''
                ln_curr_prot_w_inc = (o.ln_f[i][s] + o.ln_b[i][s] + __safe_log(curr_weight) + curr_mix_density.ln_em(o.seq[i-1])) - o.ln_eMat[s][i-1]
                
                
                #update the weight for whole observation sequence
                ln_sum_curr_prot_w = __e_ln_sum(ln_sum_curr_prot_w,ln_curr_prot_w_inc)
                #update the mean for the current mixture and observation position
                curr_prot_m_inc = __safe_exp(ln_curr_prot_w_inc)*NUM.asarray(o.seq[i-1])
                #update the mean for whole observation sequence
                sum_curr_prot_m += curr_prot_m_inc
                    
                #update the covar for the current mixture and observation position
                o.ln_gamma[(i-1,s,k)] = ln_curr_prot_w_inc
            
            # the following updates use "+=" as we are adding up all the effects from each tr_obj, 
            # as outlined in Rabiner's "Multiple Observation Sequences" pg273	
            WC[empos][k] += __safe_exp(ln_sum_curr_prot_w - o.lprob)
            MC[em_mix_pos][k] += sum_curr_prot_m/__safe_exp(o.lprob)
            '''
            if(Scale == None):
                o.prob = NUM.exp(o.lprob)
                WC[empos][k] += sum_curr_prot_w/o.prob
                MC[em_mix_pos][k] += sum_curr_prot_m/o.prob
            else:
                WC[empos][k] += sum_curr_prot_w
                MC[em_mix_pos][k] += sum_curr_prot_m
            '''
            
	# by default python passes parameters by reference to save space and be more efficient!
		
#######################################################

def gradLogP(hmm,seq,Scale=None,labels=None,multiplyEmission=None):
	'''
	gradLogP(hmm,trobj,Scale=None,labels=None,,multiplyEmission=None)
		o hmm           -> the hmm to train
		o seq           -> input sequence / or vectors
		o Scale=None    -> scaling (if != from None  scaling is used
		o labels=None   -> if labels is != None labels of the trainable objects are use
		o multiplyEmission=None -> if set != Null computes
		                        e_s(c) * gradLogP[s][c] instead of
		                        gradLogP[s][c]
		=> return gradLogP a dictionary of dictionary
		   gradLogP -> gradLogP[s][c]  = E_s(c)/e_s(c) - E_s
		   s is an emitting state and c is a sequence symbol
	'''
	dim_em_alphabet=len(hmm.emission_alphabet)
	E=hmm.emits # the list of all emitting states
	num_emit=len(E) 
	EC=NUM.array([[0.0]*dim_em_alphabet]*num_emit,ARRAYFLOAT)
	# COmpute forward and backward
	if(labels):
		(f,b,eMat,Scale,lp)=for_back_mat(hmm, seq, Scale, labels=trobj.labels)
	else:
		(f,b,eMat,Scale,lp)=for_back_mat(hmm, seq, Scale, labels=None)
	if(type(seq[0]) != type('String')): # vectors
		fT=NUM.transpose(f) # fT = transpose of forward matrix
		bT=NUM.transpose(b) # bT = transpose of backward matrix
		seqT=NUM.transpose(seq) # seqT = transpose of the sequnce vector
		if(Scale != None):
			for s in E:
				# empos= s -1 # subtract the begin staste , the emitting states start from 1
				# matrix multiplication
				EC[s-1]+=NUM.dot(seqT,fT[s][1:]*bT[s][1:]*Scale[1:])
		else:
			for s in E:
				# empos== s -1  subtract the begin staste , the emitting states start from 1
				EC[s-1]+=NUM.dot(seqT,fT[s][1:]*bT[s][1:])
				EC[s-1]/=NUM.exp(lp)
	else: # symbols
		if(Scale != None):
			for s in E:
				empos= s -1 # subtract the begin staste , the emitting states start from 1
				for c in hmm.emission_alphabet:
					cpos=hmm.emission_alphabet.index(c)
					sum_curr_prot=0.0
					for i in range(1,len(seq)+1): # from position 2 of seq to the last
						if(seq[i-1] == c):
							sum_curr_prot+=f[i][s]*b[i][s]*Scale[i]
					EC[empos][cpos]+=sum_curr_prot
		else: 
			for s in E:
				empos=s - 1  
				for c in hmm.emission_alphabet:
					cpos=hmm.emission_alphabet.index(c)
					sum_curr_prot=0.0
					for i in range(1,o.len+1): # from position 2 of seq to the last
						if(seq[i-1] == c):
							sum_curr_prot+=f[i][s]*b[i][s]
					prob=NUM.exp(lp)
					EC[empos][cpos]+=sum_curr_prot/prob
      
	# return dictionary
	grad={}
	if multiplyEmission:
		for s in E:
			sname=hmm.state_names[s]
			grad[sname]={}
			sumc=NUM.sum(EC[s-1])
			for i in range(dim_em_alphabet):
				c=hmm.emission_alphabet[i] 
				grad[sname][c]=EC[s-1][i] - sumc*hmm.e(s,c)
	else:
		for s in E:
			sname=hmm.state_names[s]
			grad[sname]={}
			sumc=NUM.sum(EC[s-1])
			for i in range(dim_em_alphabet):
				c=hmm.emission_alphabet[i] 
				grad[sname][c]=EC[s-1][i]/hmm.e(s,c) - sumc
	return(grad)

#######################################################

def Baum_Welch(hmm,set_of_trobj,pseudoAC=0.0,pseudoWC=1.0,pseudoMC=0.0,pseudoUC=1.0,Scale=None,labels=None,maxcycles=10000,tolerance=DEF.small_positive,verbose=None):
   ''' 
   Baum_Welch(hmm,set_of_trobj,Scale=None,labels=None,maxcycles=10000,tolerance=DEF.small_positive,pc=0,verbose=None)
		o hmm           -> the hmm to train
		o set_of_trobj  -> list of trainable objects
		o Scale=None    -> scaling (if != from None scaling is used
		o labels=None   -> if labels is != None labels of the trainable objects are used
		o maxcycles=10000 -> maximum number of training cycles 
		o tolerance -> we stop the learning if log(ProbNew/Probold) < tolerance
		o verbose -> print on the screen the probability every cycles
		o pseudoC -> pseudocount to add
		=> return (lPr,delta_P,cyc)
			lPr -> final log(probability)
			delta_P -> log(ProbNew/Probold)/log(ProbNew)
			cyc -> final number of cycles
   '''
   # test if we are dealing with sequence of symbols or sequence of vectors
   if(type(set_of_trobj[0].seq[0]) != type('String') ): # vectors
       #update_AC_EC=__vec_update_AC_EC
       update_AC_EC = __gmm_update
       init_AC_EC = init_AC_WC_MC_UC
       __set_param = __gmm_set_param
       if(verbose):
           #print "GIGI vector training"
           print "GMM continous emission training"
           if labels:
               print "learning with labels "
   else: # symbols
       update_AC_EC=__symbol_update_AC_EC
       if(verbose): 
           print "Classical sequence training"
           if labels:
               print "learning with labels "
   number_of_seqs=len(set_of_trobj)
   # Start
   lPtot=DEF.big_negative
   cyc=0
   delta=tolerance*2
   use_clustering = False
   while cyc < maxcycles and delta > tolerance:
       # compute forward, backward, probability and scale    
       lPcurr=0.0
       #AC,EC=init_AC_EC(hmm,pseudoC) # for the GMM version this should take values from the k-means/spectral clustering
       AC,WC,MC,UC = init_AC_EC(hmm,pseudoAC,pseudoWC,pseudoMC,pseudoUC,set_of_trobj,use_clustering)
       for i in range(len(set_of_trobj)):
           if verbose:
               print "object",i
           o=set_of_trobj[i]
           o.ln_eMat=None
           o.scale=None
           if(labels):
               (ln_f,ln_b,ln_eMat,lp)=ln_for_back_mat(hmm, o.seq, labels=o.labels)
           else:
               (ln_f,ln_b,ln_eMat,lp)=ln_for_back_mat(hmm, o.seq, labels=None)
           # set the new parameters in the objects
           o.ln_f=ln_f
           o.ln_b=ln_b
           o.ln_eMat=ln_eMat
           o.scale=Scale
           o.lprob=lp
           lPcurr+=lp
           #update_AC_EC(AC,EC,hmm,o,Scale,pseudoC)  # this uses the expected values for AC,EC then updates and normalises the model parameters for the current trobj (sequence)
           update_AC_EC(AC,WC,MC,UC,hmm,o,Scale)
       lPcurr/=number_of_seqs
       delta=(lPtot-lPcurr)/lPtot # delta change sign sincs it is divided by a lPtot<0
       if verbose:
           print "CYC",cyc
           print "log(Prob_old) ",lPtot,"log(Prob_new) ",lPcurr,"Diff ",delta
       # in the case of the generalised Expectation Maximiximisation
       # it is not guaranteed that Pcurr always >= Ptot
       if(delta>=0):
           #__set_param(hmm,AC,EC)  #discrete observation emissions (symbols)
           __set_param(hmm,set_of_trobj,AC,WC,MC,UC) # continuous observation emissions using GMM
           hmm.set_mA()
           hmm.set_mE() # This not used GMM emission parameters, the set_mE() function makes no sense for GMM as the observation emissions are considered to be continuous
           #hmm.write_for_humans('hmm-mod-'+str(cyc)) #create intermediate hmm model grammar file
       lPtot=lPcurr
       cyc=cyc+1
       #use_clustering = True
   return (lPtot,delta,cyc)

#######################################################
'''
    The following two functions, discriminative and viterbi_learning, have not been updated
    to work with GMM emission observations.
'''

def discriminative(hmm,set_of_trobj,Scale=None,maxcycles=1000,tolerance=DEF.small_positive,pseudoC=0.0,Riis=None,eta=0.001,verbose=None):
	''' 
	discriminative(hmm,set_of_trobj,Scale=None,maxcycles=1000,tolerance=DEF.small_positive,pc=0,Riis=None,verbose=None
		o hmm           -> the hmm to train
		o set_of_trobj  -> list of trainable objects
		o Scale=None    -> scaling (if != from None scaling is used
		o maxcycles=10000 -> maximum number of training cycles 
		o tolerance -> we stop the learning if log(ProbNew/Probold) < tolerance
		o pseudoC -> pseudocount to add
		o Riis=None     -> Krogh update as default. If  Riis!= None, the exponential Riis update is used
		o eta=0.001    -> learning rate for Riis or value to add in Krogh's learning
		o verbose -> print on the screen the probability every cycles
		=> return (loglike, cyc)
			lPr -> final log(p(y,x) -log(p(x))
			cyc -> final number of cycles
	'''
	# test if we updated as Riis or Krogh
	if Riis != None:
		set_param=__disc_Riis_set_param
	else:
		set_param=__disc_set_param
	# test if we are dealing with sequence of symbols or sequence of vectors
	if(type(set_of_trobj[0].seq[0]) != type('String') ): # vectors
		update_AC_EC=__vec_update_AC_EC
		if(verbose): 
			print "GIGI vector training"
	else: # symbols
		update_AC_EC=__symbol_update_AC_EC
		if(verbose): 
			print "Classical sequence training"
	number_of_seqs=len(set_of_trobj)
	# Start
	lPtot=DEF.big_negative
	cyc=0
	rmsd=logLike=tolerance+1
	while cyc < maxcycles and rmsd > tolerance:
		# compute forward, backward, probability and scale    
		lPcurrc=0.0
		lPcurrf=0.0
		ACc,ECc=init_AC_EC(hmm,pseudoC)
		ACf,ECf=init_AC_EC(hmm,pseudoC)
		for i in range(len(set_of_trobj)):
			# clamped phase
			o=set_of_trobj[i]
			(f,b,eMat,Scale,lpc)=for_back_mat(hmm, o.seq, Scale, labels=o.labels)
			# set the new parameters in the objects
			o.f=f
			o.b=b
			o.eMat=eMat
			o.scale=Scale
			o.lprob=lpc
			lPcurrc+=lpc
			update_AC_EC(ACc,ECc,hmm,o,Scale,pseudoC)
			# free phase
			(f,b,eMat,Scale,lpf)=for_back_mat(hmm, o.seq, Scale, labels=None)
			o.f=f
			o.b=b
			o.eMat=eMat
			o.scale=Scale
			o.lprob=lpf
			lPcurrf+=lpf
			update_AC_EC(ACf,ECf,hmm,o,Scale,pseudoC)
			# discharge the arrys
			o.eMat=None
			o.scale=None
			if verbose:
				print "object",i,"log clamp",lpc,"log free",lpf
		lPcurrf/=number_of_seqs
		lPcurrc/=number_of_seqs
		logLike=(lPcurrf-lPcurrc) # delta change sign sincs it is divided by a lPtot<0
		# in the case of the generalised Expectation Maximiximisation
		# it is not guaranteed that Pcurr always >= Ptot
		rmsd=set_param(hmm,ACc,ECc,ACf,ECf,eta)
		hmm.set_mA()
		hmm.set_mE()
		print "CYC",cyc
		print "log(Prob_clamped) ",lPcurrc,"log(Prob_free) ",lPcurrf,"Diff ",logLike, "RMSD parameters= ",rmsd
		# hmm.write_for_humans('hmm-mod-'+str(cyc))
		cyc=cyc+1
    
	return (logLike,cyc)
            
#######################################################

def viterbi_learning(hmm,set_of_trobj,labels=None,maxcycles=1000,tolerance=DEF.small_positive,pseudoC=0.0,verbose=None):
	''' 
	viterbi_learning(hmm,set_of_trobj,maxcycles=1000,tolerance=DEF.small_positive,pc=0,verbose=None
		o hmm           -> the hmm to train
		o set_of_trobj  -> list of trainable objects
		o labels=None   -> if labels is != None labels of the trainable objects are used
		o maxcycles=10000 -> maximum number of training cycles 
		o tolerance -> we stop the learning if log(ProbNew/Probold) < tolerance
		o verbose -> print on the screen the probability every cycles
		o pseudoC -> pseudocount to add
		=> return (lPr,abs(delta_P),cyc)
			lPr -> final log(probability)
			delta_P -> log(ProbNew/Probold)/log(ProbNew)
			cyc -> final number of cycles
	'''
	# test if we are dealing with sequence of symbols or sequence of vectors
	if(type(set_of_trobj[0].seq[0]) != type('String') ): # vectors
		if(verbose): 
			print "GIGI vector training"
			if labels:
				print "learning with labels "
	else: # symbols
		if(verbose): 
			print "Classical sequence training"
			if labels:
				print "learning with labels "
	number_of_seqs=len(set_of_trobj)
	# Start
	lPtot=DEF.big_negative
	cyc=0
	delta=tolerance*2
	while cyc < maxcycles and delta > tolerance:
		# compute forward, backward, probability and scale    
		lPcurr=0.0
		AC,EC=init_AC_EC(hmm,pseudoC)
		for i in range(len(set_of_trobj)):
			if verbose:
				print "object",i
			o=set_of_trobj[i]
			if(labels):
				(best_path,lp,lpath)=_viterbi(hmm, o.seq, labels=o.labels)
			else:
				(best_path,lp,lpath)=_viterbi(hmm, o.seq, labels=None)
			# set the new parameters in the objects
			o.lprob=lp
			lPcurr+=lp
			_update_Vit_AC_EC(AC,EC,hmm,o.seq,best_path,pseudoC)
		lPcurr/=number_of_seqs
		delta=(lPtot-lPcurr)/lPtot # delta change sign sincs it is divided by a lPtot<0
		if verbose:
			print "CYC",cyc
			print "log(Prob_old) ",lPtot,"log(Prob_new) ",lPcurr,"Diff ",delta
		# in the case of the generalised Expectation Maximiximisation
		# it is not guaranteed that Pcurr always >= Ptot
		if(delta>=0):
			__set_param(hmm,AC,EC)
			hmm.set_mA()
			hmm.set_mE()
		# hmm.write_for_humans('hmm-mod-'+str(cyc))
		lPtot=lPcurr
		cyc=cyc+1

	return (lPtot,delta,cyc)
