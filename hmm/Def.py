'''
Some definitions used all through the HMM
'''
class DEF:
   ''' contains definitions used
       everywhere in the hmm codes
   '''
   big_negative=-1000000000.0 # maximum negative number instead of log(0) 
   big_positive=1000000000.0 # maximum positive number instead of letting numerical inf occur
   small_positive=1e-30
   tolerance=small_positive # if > tolerance we can take the log
   # alpha: positive float --> Sparse Inverse Covariance Estimation from Sklearn.covariance.graph_lasso
   # The regularization parameter: the higher alpha, the more regularization, 
   # the sparser the inverse covariance
   alpha = 0.25
   min_covar=1.e-6 #used for regularization of the covariance matrix in multivariate_gaussian
