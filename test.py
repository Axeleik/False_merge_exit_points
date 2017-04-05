import h5py
import numpy as np
import copy
import pickle
from functions import comparison

import vigra





def printname(name):
    print name

if __name__ == "__main__":
    pass
    f = h5py.File("/media/axeleik/EA62ECB562EC8821/data/ground_truth.h5", mode='r')
    g = h5py.File("/media/axeleik/EA62ECB562EC8821/data/result_resolved.h5", mode='r')
    h = h5py.File("/media/axeleik/EA62ECB562EC8821/data/result.h5", mode='r')



    ground_truth = np.array(f["z/1/neuron_ids"])
    segmentation = np.array(h["z/1/test"])
    segmentation_resolved = np.array(g["z/1/test"])


    bc = comparison(segmentation,ground_truth,segmentation_resolved)

    print "bc: ",bc,"\n\n","Length: ",len(bc)
    x=0
    for i in bc.keys():
        if bc[i] == False:
            print i,": ", "False"
            x=x+1

    print "Length_False: ", x





