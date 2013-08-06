#!/usr/bin/env python

import sys
import string
import math
import re


def readPreds(fname):
    ''' readPreds(fname) '''
    try:
        lines=open(fname,'r').readlines()
    except:
        sys.stderr.write('cannot open file '+fname+'\n')
        sys.exit()
    v=lines[0].split() # assuming first line # pos obs hmm_apv match
    P={}
    names=[]
    for e in v[1:-2]:  # # pos obs hmm_apv match
        P[e]=''
        names.append(e)
    for line in lines[1:]:
        v=line.split()
        for i in range(len(names)):
            P[names[i]]+=v[i+1]
    return P,names
    
def cmatrix(obs,pred):
    ''' cmatrix(obs,pred) '''
    cm={}
    for o,p in zip(obs,pred):
        cm[(o,p)]=cm.get((o,p),0)+1
    return cm    

def getSegs(ss, symbol='H'):
    ''' getSegs(ss, symbol='H') '''
    segments=[]
    pat=re.compile(symbol+'+')
    groupSS=pat.search(ss)
    while groupSS:
         b,e=groupSS.span()
         segments.append((b,e))
         groupSS=pat.search(ss,e)
    return segments
    
def averOv(s1,s2):
    ''' averOv(s1,s2) '''
    b1,e1=s1
    b2,e2=s2
    ov=0
    if b1 < e2 and b2 < e1:
        e=min(e1,e2)
        b=max(b1,b2)
        ov=e-b
    if ov > 0.25*((e1-b1)+(e2-b2)): # half of average segment 1/2 * 1/2
        return 1
    else:
        return 0

def fixOv(s1,s2,Th=2):
    ''' fixOv(s1,s2,Th=2) '''
    b1,e1=s1
    b2,e2=s2
    ov=0
    if b1 < e2 and b2 < e1:
        e=min(e1,e2)
        b=max(b1,b2)
        ov=e-b
    if ov >=Th:
        return 1
    else:
        return 0

def minOv(s1,s2):
    ''' minOv(s1,s2) '''
    b1,e1=s1
    b2,e2=s2
    ov=0
    if b1 < e2 and b2 < e1:
        e=min(e1,e2)
        b=max(b1,b2)
        ov=e-b
    if ov >= 0.5* min((e1-b1),(e2-b2)):
        return 1
    else:
        return 0

def nterminus(ss,symbols=['b','f']):
    '''  nterminus(ss,symbols=['b','f'])'''
    if ss[0] in symbols:
        return ss[0]
    else: # start with H the first met is the second type
        i=ss.find(symbols[0])
        o=ss.find(symbols[1])
        if i < 0:
            return symbols[0]
        elif o < 0: 
            return symbols[1]
        elif i < o :  
            return symbols[1]
        else:
            return symbols[0]
    

def XProt(obs,pred,segOverlap=fixOv):
    '''  XProt(obs,pred) '''
    oseg=getSegs(obs,'b')
    pseg=getSegs(pred,'b')
    if len(oseg) != len(pseg) :
        return 0,0,"different number of segments"  # wrong topology and wrong topography
    # check overlap
    segOk=1
    for os,ps in zip(oseg,pseg):
        if not segOverlap(os,ps):
            return 0,0,"wrong overlap"  # wrong topology and wrong topography
    if nterminus(obs) == nterminus(pred):
        return 1,1,"correct topology" 
    else:    
        return 1,0,"wrong topology correct topography" 
     

def xRes(cm):
    ''' xRes(cm) '''
    pclass=[]
    for o,p in cm.keys():
        if o not in pclass:
           pclass.append(o)
    Nclass=len(pclass)
    Spec={}
    Sens={}
    C={}
    Q=0.0
    NQ=0.0
    for o in pclass:
        nok=cm.get((o,o),0.0)
        Q+=nok
        sen=spn=0.0
        for p in pclass:
            sen+=cm.get((o,p),0.0)
            spn+=cm.get((p,o),0.0)
        NQ+=spn    
        try:
            Sens[o]=nok/sen
        except:
            Sens[o]=None
        try:    
            Spec[o]=nok/spn
        except:
            Spec[o]= None
        cset=set(pclass)
        cset.remove(o)
        pok=ue=oe=0.0
        for p in cset:
            ue+=cm.get((o,p),0.0)
            oe+=cm.get((p,o),0.0)
            for p2 in cset:
                pok+=cm.get((p2,p),0.0)
        try:
            C[o]=(pok*nok - ue*oe)/math.sqrt((pok+ue)*(pok+oe)*(nok+ue)*(nok+oe))
        except:
            C[o]=None
    try:
        Q/=NQ
    except:
        Q=None
    return pclass,Q,Spec,Sens,C  
#----------------MAIN-----------------
if __name__=='__main__':
    if len(sys.argv) < 1:
        print "usage :",sys.argv[0]," predfile"
        sys.exit()
    DP,names=readPreds(sys.argv[1]+'.apv.pred') 
    print 'X residue'
    for n in names[1:]:
        print "#XR",n,
#        print cmatrix(DP[names[0]],DP[n])
        pclass,Q,Spec,Sens,C=xRes(cmatrix(DP[names[0]],DP[n])) 
        print 'Q=',Q,
        for c in pclass:
            print c, 'Sp=',Spec[c],'Sen=',Sens[c],'Corr=',C[c], 
        print

    print 'X protein'
    for n in names[1:]:
        print "#XP",n,
        tmpl=XProt(DP[names[0]],DP[n]) 
        for e in tmpl:
            print e,
        print

