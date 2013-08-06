import sys

def main():
    if len(sys.argv)<2:
        print "usage :",sys.argv[0],"ProtTop.results, ProtTopEval.results"
        sys.exit(-1)
    
    ifname = sys.argv[1]
    ofname = sys.argv[2]
    iPTfile = open(ifname,'r')
    
    toplines = iPTfile.readlines()
    
    iPTfile.close()
    
    num_lines = 0
    tot_num = 0.0
    num_diff_seg = 0.0 # different segments, wrong topology and wrong topography
    num_correct_top = 0.0 # correct topology
    num_wrong_top = 0.0 # wrong topology and correct topography
    for line in toplines:
        elems = line.rstrip().split()
        #print elems
        num_lines += 1
        if int(elems[0]) == 0 and int(elems[1]) == 1:
            num_diff_seg += 1
        elif int(elems[0]) == 1 and int(elems[1]) == 1:
            num_correct_top += 1
        else:
            num_wrong_top += 1
    
    tot_num = num_wrong_top + num_correct_top + num_diff_seg
    print "Num Proteins: %s"%num_lines
    out_str = "Total Num Proteins: %s, \nPercentage wrong topology: %s, \nPercentage different segments: %s, \nPercentage correct topology: %s \n"%(tot_num,num_wrong_top/tot_num,num_diff_seg/tot_num,num_correct_top/tot_num)
    oPTfile = open(ofname,"w")     
    oPTfile.write(out_str)
    oPTfile.close()
    
if __name__ == '__main__':
    main()