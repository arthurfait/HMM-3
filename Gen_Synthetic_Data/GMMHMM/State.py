'''
This file contains the definitions of:

node_tr => transitions class 
node_em => emission class
node_em_gmm => gaussian mixture model emission class (derived class of node_em)
State => state class which contains 
            one pointer to a node_tr object
            one pointer to a node_em object


Extending the code of Piero Fariselli to include the GMM for each state, aka nodes.
When GMM are used the node_em class, emission class, must be overloaded/over-ridden 
to rather utilize the Gaussian Mixtures, node_em_gmm.
'''

from Def import DEF
import numpy as NUM
import sys
from sklearn.covariance import graph_lasso

######################################
######## class node_trans ############
######################################

class node_tr:
    ''' 
        this class implement the node transitions
    '''
    def __init__(self,name,tr):
        '''
            __init__(self,name,tr) 
            name = identifier, tr = transition probabilities 
        '''
        self.name=name  # "esempio '_tr1','_tr2' ...  
        self.len=len(tr)
        self._tr=tr # Vettore di transizioni
        # computes the log of the transitions
        self._ln_tr=[DEF.big_negative]*self.len
        for i in range(self.len):
            self._ln_tr[i] = self.__safe_log(self._tr[i])
            #if(self._tr[i] > DEF.tolerance):
            #   self._ln_tr[i]=NUM.log(self._tr[i])
    
    def __safe_log(self,x):
        if x < DEF.small_positive:
            return DEF.big_negative
        else:
            return NUM.log(x)
        
    def tr(self,i):
        ''' 
            tr(self,i) -> transition probability between self->i-th
        '''
        #assert(i<self.len)
        return self._tr[i]
    
    def ln_tr(self,i):
        ''' 
            ln(tr(self,i)) -> transition probability between self->i-th
        '''
        #assert(i<self.len)
        return self._ln_tr[i]
    
    def set_tr(self,i,value):
        '''
            set_tr(self,i,value) sets the i-th transition of self to value
        '''
        #assert(i<self.len)
        self._tr[i]=value
        if(self._tr[i] > DEF.tolerance):
            self._ln_tr[i]=NUM.log(self._tr[i])
        else:
            self._ln_tr[i]=DEF.big_negative
            
######################################
######################################
######## class node_em    ############
######################################
class node_em:
    ''' 
        this class implement the node emissions and is a base class for discrete emissions and gmm
    '''
    def __init__(self,name,em):
        '''
            __init__(self,name,em=None)
            name = identifier, em = emission probabilities
        '''
        self.name=name  # "esempio '_tr1','_tr2' ...  
        self.len=len(em) # dimensione del vettore
        self._em=em # Vettore emissioni
        # computes the log of the emissions 
        self._ln_em=[DEF.big_negative]*self.len
        for i in range(self.len):
            if(self._em[i] > DEF.tolerance):
                self._ln_em[i]=NUM.log(self._em[i])
    
    def __safe_log(self,x):
        if x < DEF.small_positive:
            return DEF.big_negative
        else:
            return NUM.log(x)
        
    def em(self,i):
        '''
            em(self,i) -> emission probability of discrete symbol i in the current node
        '''
        #assert(i<self.len)
        return self._em[i]
    
    def ln_em(self,i):
        '''
            em(self,i) -> emission probability of discrete symbol i in the current node
        '''
        #assert(i<self.len)
        return self._ln_em[i]
    
    def normalise_mix_weights(self):
        '''
            normalise the mixture weights for this state so that they sum to one.
            This is for the GMM version, which is accessed via dynamic binding and
            polymorphism.
        '''
        return True
    
    def get_emissions(self):
        '''
            Returns the list of discrete emission probabilities
        '''
        return self._em
    
    def get_type_name(self):
        '''
            returns the type name for this node emission object. This is used for setting the precomputed emissions matrix.
        '''
        return "node_em"

######################################
######################################
######## class node_gmm    ############
######################################
class node_em_gmm(node_em):
    '''
        this class implements the node emissions using gaussian mixture models
    '''
    def __init__(self,name,mix_weights,mix_densities):
        ''' 
            __init__(self,name,em=None)
            name = identifier,
            mix_weights = list of weights for the mixtures,
            mix_densities = list of mixture_density objects
        '''
        self.name=name  # "esempio '_tr1','_tr2' ...
        self._mix_weights = mix_weights
        self._mix_num = len(self._mix_weights)
        self._mix_densities = mix_densities  # In the literature this is referred to as a codebook
        if self._mix_num > 0:
            self._len = len(self._mix_densities[0].get_mean()) # shape/dim of the GMM parameters
        else:
            self._len = 0  #no mixtures in a null/silent state
    
    def __safe_log(self,x):
        if x < DEF.small_positive:
            return DEF.big_negative
        else:
            return NUM.log(x)
        
    def get_mix_num(self):
        '''
            mix_num(self) -> number of gaussian mixtures for this node
        '''
        #assert(i<self.len)
        return self._mix_num
    
    def get_mix_weight(self,i):
        '''
            mix_weight(self,i) -> returns the i-th mixture's weight for this node
        '''
        #assert((i>=0) and (i<self._mix_num)) # assert the index of the mixture weight
        return self._mix_weights[i]

    def get_mixture_density(self,i):
        '''
            Returns a mixture_density object with corresponding parameters
        '''
        return self._mix_densities[i]
    
    def get_em_mix_name(self):
        '''
            Returns the name of the mixture densities for this state
        '''
        return self._mix_densities[0].get_name() 
        
    def get_mixtures(self):
        '''
            Returns a reference to this state's list of mixture densities
        '''
        return self._mix_densities
    
    def get_emissions(self):
        '''
            returns the node_em_gmm object instance
        '''
        return self
    
    def set_mix_num(self,value):
        '''
            set_mix_num(self,value) set the number of mixtures for this node
        '''
        #assert(value >= 0)
        self._mix_num=int(value)
    
    def set_mix_weight(self,i,weight):
        '''
            set the i-th mixture's weight value for this node
        '''
        if ((i>=0) and (i < self._mix_num)):
            self._mix_weights[i] = float(weight)
        else:
            print(self._name + ": cannot set mixture " + str(i) + " weight.\n")
    
    def set_mix_density(self,i,mix_density):
        '''
            Set the i-th mixture's density function with mean, covariance, and precision matrices
        '''
        self._mix_densities[i] = mix_density
    
    def normalise_mix_weights(self):
        '''
            normalise the mixture weights for this state so that they sum to one.
        '''
        err_tol = 6 #five decimal places for precision
        sum_val = 1.0 #the value which the mixture weights must sum too
        weight_sum = float(sum(self._mix_weights))
        if (NUM.absolute(NUM.round(weight_sum,err_tol) - sum_val) < DEF.tolerance):
            self._mix_weights = [w/weight_sum for w in self._mix_weights]
        #return lets the user know if the weight were already normalised or not
        return (NUM.absolute(NUM.round(weight_sum,err_tol) - sum_val) < DEF.tolerance)
    
    def em(self,vec):
        '''
            em(self,vec) -> compute the probability of emission of vec in the node/state
            with the mixtures. The default option is the multivariate gaussians, but any
            pdf with the required properties will do.
        '''
        lprob = self.ln_em(vec)
        prob = NUM.exp(lprob)
        return prob
    
    def ln_em(self,vec):
        '''
            ln_em(self,vec) -> compute the ln of probability of emission of vec in the node/state
            with the gaussian mixtures.
        '''
        assert(len(vec) == self._len)
        lprob = 0.0
        for i in range(self._mix_num):
            linc = self.__safe_log(self._mix_weights[i])+self._mix_densities[i].ln_em(vec)
            lprob += linc
        return lprob
    
    def get_type_name(self):
        '''
            returns the type name for this node emission object. This is used for setting the precomputed emissions matrix.
        '''
        return "node_em_gmm"

######################################

class mixture_density:
    '''
        A class represent the mixture densities used for the hidden state emissions.
    '''
    def __init__(self,name,mean,covariance,precision=None):
        self._name = name
        self._mean = mean
        self._cov = covariance
        self._precision = precision
        #compute the mix_precisions using spare inverse covariance estimation by graph lasso method
        # must use sparse inverse covariance estimation by graphical lasso method as the determinant of a 
        # non-diagonal covariance matrix will be "too small" and cause numerical issues
        # These are set during the Baum-Welch learning phase in the set_param() function
        '''        
        for i in range(self._mix_num):
            #compute and set the covariance precision matrices corresponding to the newly updated empirical covariance matrices
            # here we use the empirical covariance as input to the graphlasso algorithm, with a predefined
            # alpha: positive float
            # The regularization parameter: the higher alpha, the more regularization, the sparser the inverse covariance
            emp_cov = self._mix_covars[i]            
            glasso_cov, glasso_precision = graph_lasso(emp_cov, DEF.alpha, mode='cd', tol=1e-4, max_iter=100,verbose=False)    
            self._mix_covars[i] = glasso_cov
            self._mix_precisions[i] = glasso_precision
        '''
    
    def __safe_log(self,x):
        if x < DEF.small_positive:
            return DEF.big_negative
        else:
            return NUM.log(x)
        
    def get_name(self):
        return self._name
    
    def get_mean(self):
        return self._mean
    
    def get_cov(self):
        return self._cov
    
    def get_precision(self):
        return self._precision
    
    def set_name(self,name):
        self._name = name
    
    def set_mean(self,mean):
        self._mean = mean
    
    def set_cov(self,covariance):
        self._cov = covariance
    
    def set_precision(self,precision):
        self._precision = precision
    
    def em(self,vec):
        prob = self.multivariate_gaussian(vec)
        return prob
    
    def ln_em(self,vec):
        ln_prob = self.ln_multivariate_gaussian(vec)
        return ln_prob
    
    def multivariate_gaussian(self,vec):
        '''
            multi_gaussian(self,vec,mean,covar,precision) -> compute the probability for a vec
            using the multivariate gaussian with mean, covar and precision.
            
            This only works on positive semi-definite matrices.
            If the variance-covariance matrix is semi-definite then one of the random
            variables is a linear combination of the others and is therefore degenerate.
        '''
        result = 0.0
        dim = len(self._mean)
        det_covar = NUM.linalg.det(self._cov)
        if ((len(vec) == dim) and ((dim,dim) == self._cov.shape)):
            if (det_covar == 0):
                sys.stderr.write("singular covariance matrix")
                sys.exit(-1)
            else:
                if self._precision == None:
                    self._precision = NUM.matrix(NUM.linalg.pinv(self._cov))
                vec_mean = NUM.matrix(NUM.matrix(vec) - NUM.matrix(self._mean))
                #frac = NUM.power(2.0*NUM.pi,0.5*dim)*NUM.power(det_covar,0.5)
                part1 = -0.5*dim*self.__safe_log(2.0*NUM.pi)
                part2 = -0.5*self.__safe_log(det_covar)
                part3 = -0.5*float(((vec_mean*self._precision)*vec_mean.T))
                log_result = part1 + part2 + part3
                result = NUM.exp(log_result)
        else:
            sys.stderr.write("The dimensions of the input don't match.")
            sys.exit(-1)
        #if result == 0.0:
        #    print "prob %f"%result
        #    print self._cov, vec
        #    print "Determinant of Covar %f"%det_covar
        #    print "Part1 %f, Part2 %f, Part3 %f"%(part1,part2,part3)
        return result
    
    def ln_multivariate_gaussian(self,vec):
        '''
        multi_gaussian(self,vec,mean,covar,precision) -> compute the probability for a vec
        using the multivariate gaussian with mean, covar and precision.
            
        This only works on positive semi-definite matrices.
        If the variance-covariance matrix is semi-definite then one of the random
        variables is a linear combination of the others and is therefore degenerate.
        '''
        result = 0.0
        dim = len(self._mean)
        det_covar = NUM.linalg.det(self._cov)
        if ((len(vec) == dim) and ((dim,dim) == self._cov.shape)):
            if (det_covar == 0):
                sys.stderr.write("singular covariance matrix")
                sys.exit(-1)
            else:
                if self._precision == None:
                    self._precision = NUM.matrix(NUM.linalg.pinv(self._cov))
                vec_mean = NUM.matrix(NUM.matrix(vec) - NUM.matrix(self._mean))
                #frac = NUM.power(2.0*NUM.pi,0.5*dim)*NUM.power(det_covar,0.5)
                part1 = -0.5*dim*self.__safe_log(2.0*NUM.pi)
                part2 = -0.5*self.__safe_log(det_covar)
                part3 = -0.5*float(((vec_mean*self._precision)*vec_mean.T))
                log_result = part1 + part2 + part3
                #result = NUM.exp(log_result)
                result = log_result
        else:
            sys.stderr.write("The dimensions of the input don't match.")
            sys.exit(-1)
        #if result == 0.0:
        #    print "prob %f"%result
        #    print self._cov, vec
        #    print "Determinant of Covar %f"%det_covar
        #    print "Part1 %f, Part2 %f, Part3 %f"%(part1,part2,part3)
        return result

######################################
######## class State      ############
######################################
class State:
    '''
        This class implements the state of a HMM
    '''
    def __init__(self,name,n_tr,n_em,out_s,in_s,em_let,tied_t,tied_e,tied_e_mix,end_s,label=None):
        '''
            __init__(self,name,n_tr,n_em,out_s,in_s,em_let,tied_t,tied_e,end_s,label=None)
            name = state name
            n_tr = a node_tr object
            n_em = a node_em object (either discrete symbol or GMM)
            out_s = the state outlinks [list of the state names] 
            in_s = the state inlinks [list of the state names]
            em_let = emission letter [list in the order given by n_em]
            tied_t = if is tied to a given transition (state name or None) 
            tied_e = if is tied to a given emission at state level (state name or None)
            tied_e_mix = if is tied to a given state's list of mixture densities at sub-state mixture-tying level (state name or None)
            end_s = end state flag
            label = classification attribute (None default)
            _idxem={} dictionary name:index 
            _idxtr={} dictionary name:index
        '''
        self.name=name
        self._node_tr=n_tr
        self._node_em=n_em
        self.out_links=out_s
        self.in_links=in_s
        self.em_letters=em_let
        self.tied_t=tied_t
        self.tied_e=tied_e
        self.tied_e_mix = tied_e_mix
        self.end_state=end_s
        self.label=label
        self._idxem={}
        self._idxtr={}
        for name in self.out_links:
            self._idxtr[name]=self.out_links.index(name)
        for symbol in self.em_letters:
            self._idxem[symbol]=self.em_letters.index(symbol)
        # some tests
        #assert(self._node_tr.len == len(self.out_links))
        #assert(self._node_em.len == len(self.em_letters))
        #check if state is null/silent
        if (self._node_em.get_type_name() == "node_em_gmm"):
            if (self._node_em.get_mix_num() == 0):
                self.silent = True
            else:
                self.silent = False
        else:
            if (self.em_letters == []):
                self.silent = True
            else:
                self.silent = False

    def __safe_log(self,x):
        if x < DEF.small_positive:
            return DEF.big_negative
        else:
            return NUM.log(x)
        
    def get_tr_name(self):
        '''
            get_tr_name() -> returns the name of the transitions
        '''
        return self._node_tr.name
    
    def get_em_name(self):
        '''
            get_em_name() -> returns the name of the emissions
        '''
        return self._node_em.name
    
    def get_em_mix_name(self):
        '''
            Returns the name of emission mixture densities. Since states can only share all or none of their mixtures
            this is the same for all mixture densities in a state
        '''
        return self._node_em.get_em_mix_name()
    
    def get_transitions(self):
        '''
            get_transitions() -> returns the value of the transitions
        '''
        return self._node_tr._tr
    
    def get_emissions(self):
        '''
            get_emissions() -> if the node_em object is the discrete symbol version
                then return a list of emission probs else if the gmm version
                then returns the node_em_gmm object with the number of 
                mixtures, a list of means, and a list of covariance matrices
        '''
        return self._node_em.get_emissions()
    
    def a(self,state):
        '''
            a_{i,j} in durbin et al., 1998
            self.a(state)  -> transition probability between self->state
        '''
        if self._idxtr.has_key(state.name):
            return self._node_tr._tr[self._idxtr[state.name]]
        else:
            return(0.0)
    
    def set_a(self,state,value):
        '''
            set the value of a_{i,j} in durbin et al., 1998
            self.a(state,value)  -> self->state = value
        '''
        self._node_tr.set_tr(self._idxtr[state.name],value)
    
    def e(self,symbol):
        ''' 
            e_{k}(x) in durbin et al., 1998
            self.e(symbol)  -> emission probability in state self of  'symbol' 
        '''
        if (len(symbol) == 1):
            if self._idxem.has_key(symbol):
                return self._node_em.em(self._idxem[symbol])
            else:
                return(0.0)
        else:
            return self._node_em.em(symbol) #symbol is actually a vector profile in this case
    
    def set_e(self,symbol,value):
        '''
            set the value of e_{k}(x) in durbin et al., 1998
            self.e(symbol,value)  -> set self.e(symbol)=value 
        '''
        self._node_em.set_em(self._idxem[symbol],value)
    
    def ln_a(self,state):
        '''
            ln(a_{i,j}) in durbin et al., 1998
            self.ln_a(state)  -> log(transition probability between self->state)
        '''
        if self._idxtr.has_key(state.name):
            return(self._node_tr.ln_tr(self._idxtr[state.name]))
        else:
            return(DEF.big_negative)
    
    def ln_e(self,symbol):
        '''
            ln(e_{k}(x)) in durbin et al., 1998
            self.ln_e(symbol)  -> log(emission probability in state self of  'symbol') 
        '''
        #ce=self.e(symbol)
        #if ce > 0:
        #    return(NUM.log(ce))
        #else:
        #    return(DEF.big_negative)
        if (len(symbol) == 1):
            if self._idxem.has_key(symbol):
                return self._node_em.ln_em(self._idxem[symbol])
            else:
                return(0.0)
        else:
            return self._node_em.ln_em(symbol) #symbol is actually a vector profile in this case
    
    def is_null(self):
        '''
            Returns True or False boolean value depending on whether or not the state is
            null/silent or emitting.
        '''
        return self.silent
    
    def get_type_name(self):
        '''
            Returns the type emission node object this State node contains.
            Either node_em or node_em_gmm
        '''
        return self.get_emissions().get_type_name()
########################

