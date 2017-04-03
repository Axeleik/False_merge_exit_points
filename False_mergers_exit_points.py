import h5py
import numpy as np
import copy
import pickle
from functions import compute_border_contacts
import vigra

def printname(name):
    print name

if __name__ == "__main__":
    pass
    f = h5py.File("/media/axeleik/EA62ECB562EC8821/data/ground_truth.h5", mode='r')
    g = h5py.File("/media/axeleik/EA62ECB562EC8821/data/result_resolved.h5", mode='r')

    ground_truth = np.array(f["z/1/neuron_ids"])
    segmentation = np.array(g["z/1/test"])

    edge_volume = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmentation[:, :, z])[:, :, None] for z in
         xrange(segmentation.shape[2])],
        axis=2)
    dt = vigra.filters.distanceTransform(edge_volume, pixel_pitch=[1., 1., 10.], background=True)

    bc = compute_border_contacts(segmentation, dt)


    array=[]
    for i in bc:
        if len(bc[i])<2:
            continue
        count.append(i)
        compare=[]
        for x in bc[i]:
            compare.append(ground_truth[x[0],x[1],x[2]])
        print "compare: ", compare
        print "unique:",len(np.unique(compare))
        if len(np.unique(compare))>1:
            array.append([i,"False"])

        else:
            array.append([i, "True"])






