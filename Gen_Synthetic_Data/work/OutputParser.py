import sys

def main():
    if len(sys.argv)<2:
        print "usage :",sys.argv[0],"Eval file, Results Output Directory"
        sys.exit(-1)
    
    filename = sys.argv[1]
    outputdir = sys.argv[2]
    evalfile = open(filename,'r')
    
    line = evalfile.readline() #X Residue line
    line = evalfile.readline() #Per residue performance metrics
    '''
    elems = line.rstrip().split()
    Qfile = open(outputdir+"Q.results","a")     
    Qfile.write(elems[3]+"\n")
    Qfile.close()
    num_labels = len(elems[4:])/7 #7 is the number of elements taken up in the list for one label's eval
    for i in range(num_labels):
        f = open(outputdir+elems[4+i*7]+".results","a")
        f.write(elems[6+i*7] + "\t" + elems[8+i*7] + "\t" +elems[10+i*7]+"\n")
        # Sp Sen Corr
        f.close()
    '''
    line = evalfile.readline() #X Protein line
    line = evalfile.readline() #Per Protein performance metrics topology + topography
    elems = line.rstrip().split()
    PTfile = open(outputdir+"ProtTop.results","a")     
    PTfile.write(elems[2] +"\t"+elems[3]+"\n")
    PTfile.close()
    
if __name__ == '__main__':
    main()