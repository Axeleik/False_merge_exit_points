import h5py
import numpy as np
import copy
import pickle
#from functions import compute_border_contacts
from neu import comparison

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

    edge_volume_segmentation = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmentation[:, :, z])[:, :, None] for z in
         xrange(segmentation_resolved.shape[2])],
        axis=2)
    dt_segmentation = vigra.filters.distanceTransform(edge_volume_segmentation, pixel_pitch=[1., 1., 10.], background=True)

    print "shape: ", ground_truth.shape
    bc = comparison(segmentation, dt_segmentation,ground_truth,segmentation_resolved)

    print "bc: ",len(bc)



