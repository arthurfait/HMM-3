'''
This file contains the class trainable object (TR_OBJ).
It consists of the type of the object we want to train with 
our HMM
'''

import math

##########################################
class TR_OBJ:
	''' define a trainable object
		a TR_OBJ object cositst of:
		                               
		self.len 	# length of the sequence
		self.seq	# the list of symbols || vectors
		self.labels	# the list of the labels
		self.f		# the forward matrix
		self.b		# the backward matrix
		self.eMat       # the precalculate emission matrix
		self.scale 	# the scale factors
           self.gamma # dictionary of current cycle parameter computed values --> key = (position,state,mixture) = val
		self.prob	# the probability of the sequence
		self.lprob	# the log(probability) of the sequence
		self.name	# the object name
        
		for the GMM version the eMat is computed by the 
		mixture number, mixture weights, mixture means, and mixture covariances.                                                                         
	'''
	def __init__(self,seq,labels=[],ln_f=[],ln_b=[],ln_eMat=[],scale=[],ln_p=None,name=None):
         ''' __init__(self,seq,labels=[],f=[],b=[],scale=[],p=None) 
         self.seq	# the list of symbols || vectors
         self.labels	# the list of the labels
         self.f		# the forward matrix
         self.b		# the backward matrix
         self.scale 	# the scale factors
         self.prob	# the probability of the sequence
         self.lprob	# the log(probability) of the sequence
         self.name	# the object name
         self.gamma
         '''
         self.seq=seq		# the list of symbols || vectors
         self.len=len(seq) 	# length of the sequence
         self.labels=labels	# the list of the labels
         self.ln_f=ln_f		# the forward matrix
         self.ln_b=ln_b		# the backward matrix
         self.ln_eMat=ln_eMat          # the precalculate emission matrix
         self.scale=scale 	# the scale factors
         self.lprob=ln_p
         self.ln_gamma = {}
         self.name=name		# the object name
    

