import numpy as np
import vigra

def get_faces_with_neighbors(image):

    # --- XY ---
    # w = x + 2*z, h = y + 2*z
    shpxy = (image.shape[0] + 2*image.shape[2], image.shape[1] + 2*image.shape[2])
    xy0 = (0, 0)
    xy1 = (image.shape[2],) * 2
    xy2 = (image.shape[2] + image.shape[0], image.shape[2] + image.shape[1])
    print shpxy, xy0, xy1, xy2

    # xy front face
    xyf = np.zeros(shpxy)
    xyf[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, 0]
    xyf[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[0, :, :]), 0, 1)
    xyf[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(image[-1, :, :], 0, 1)
    xyf[xy1[0]:xy2[0], 0:xy1[1]] = np.fliplr(image[:, 0, :])
    xyf[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = image[:, -1, :]

    # xy back face
    xyb = np.zeros(shpxy)
    xyb[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, -1]
    xyb[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(image[0, :, :], 0, 1)
    xyb[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[-1, :, :]), 0, 1)
    xyb[xy1[0]:xy2[0], 0:xy1[1]] = image[:, 0, :]
    xyb[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = np.fliplr(image[:, -1, :])

    # --- XZ ---
    # w = x + 2*y, h = z + 2*y
    shpxz = (image.shape[0] + 2*image.shape[1], image.shape[2] + 2*image.shape[1])
    xz0 = (0, 0)
    xz1 = (image.shape[1],) * 2
    xz2 = (image.shape[1] + image.shape[0], image.shape[1] + image.shape[2])
    print shpxz, xz0, xz1, xz2

    # xz front face
    xzf = np.zeros(shpxz)
    xzf[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, 0, :]
    xzf[0:xz1[0], xz1[1]:xz2[1]] = np.flipud(image[0, :, :])
    xzf[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = image[-1, :, :]
    xzf[xz1[0]:xz2[0], 0:xz1[1]] = np.fliplr(image[:, :, 0])
    xzf[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = image[:, :, -1]

    # xz back face
    xzb = np.zeros(shpxz)
    xzb[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, -1, :]
    xzb[0:xz1[0], xz1[1]:xz2[1]] = image[0, :, :]
    xzb[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = np.flipud(image[-1, :, :])
    xzb[xz1[0]:xz2[0], 0:xz1[1]] = image[:, :, 0]
    xzb[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = np.fliplr(image[:, :, -1])

    # --- YZ ---
    # w = y + 2*x, h = z + 2*x
    shpyz = (image.shape[1] + 2*image.shape[0], image.shape[2] + 2*image.shape[0])
    yz0 = (0, 0)
    yz1 = (image.shape[0],) * 2
    yz2 = (image.shape[0] + image.shape[1], image.shape[0] + image.shape[2])
    print shpyz, yz0, yz1, yz2

    # yz front face
    yzf = np.zeros(shpyz)
    yzf[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[0, :, :]
    yzf[0:yz1[0], yz1[1]:yz2[1]] = np.flipud(image[:, 0, :])
    yzf[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = image[:, -1, :]
    yzf[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(np.flipud(image[:, :, 0]), 0, 1)
    yzf[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(image[:, :, -1], 0, 1)

    # yz back face
    yzb = np.zeros(shpyz)
    yzb[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[-1, :, :]
    yzb[0:yz1[0], yz1[1]:yz2[1]] = image[:, 0, :]
    yzb[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = np.flipud(image[:, -1, :])
    yzb[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(image[:, :, 0], 0, 1)
    yzb[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(np.flipud(image[:, :, -1]), 0, 1)

    faces = {
        'xyf': xyf,
        'xyb': xyb,
        'xzf': xzf,
        'xzb': xzb,
        'yzf': yzf,
        'yzb': yzb
    }

    shp = image.shape
    bounds = {
        'xyf': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
        'xyb': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
        'xzf': np.s_[shp[1]:shp[1] + shp[0], shp[1]+1:shp[1] + shp[2]-1],
        'xzb': np.s_[shp[1]:shp[1] + shp[0], shp[1]+1:shp[1] + shp[2]-1],
        'yzf': np.s_[shp[0]+1:shp[0] + shp[1]-1, shp[0]+1:shp[0] + shp[2]-1],
        'yzb': np.s_[shp[0]+1:shp[0] + shp[1]-1, shp[0]+1:shp[0] + shp[2]-1]
    }

    return faces, bounds


def find_centroids(seg, dt, bounds,centroids):



    for lbl in np.unique(seg[bounds])[1:]:

        # Mask the segmentation
        mask = seg == lbl

        # Connected component analysis to detect when a label touches the border multiple times
        conncomp = vigra.analysis.labelImageWithBackground(mask.astype(np.uint32), neighborhood=8,
                                                           background_value=0)

        # Only these labels will be used for further processing
        # FIXME expose radius as parameter
        opened_labels = np.unique(vigra.filters.discOpening(conncomp.astype(np.uint8), 2))


        # unopened_labels = np.unique(conncomp)
        # print 'opened_labels = {}'.format(opened_labels)
        # print 'unopened_labels = {}'.format(unopened_labels)
        for l in opened_labels[1:]:

            # Get the current label object
            curobj = conncomp == l

            # Get disttancetransf of the object
            cur_dt = np.array(dt)
            cur_dt[curobj == False] = 0


            # Detect the global maximum of this object
            amax = np.amax(cur_dt)
            cur_dt[cur_dt < amax] = 0
            cur_dt[cur_dt > 0] = lbl

            # Get the coordinates of the maximum pixel(s)
            coords = np.where(cur_dt[bounds])

            # If something was found
            if coords[0].any():
                # Only one pixel is allowed to be selected
                # FIXME: This may cause a bug if two maximum pixels exist that are not adjacent (although it is very unlikely)
                coords = [int(np.mean(x)) for x in coords]

                if lbl in centroids.keys():
                    centroids[lbl].append(curobj[bounds])

                else:
                    centroids[lbl] = [curobj[bounds]]


    return centroids


def compute_border_contacts(
        segmentation,
        disttransf
):

    centroids={}

    faces_seg, bounds = get_faces_with_neighbors(segmentation)
    faces_dt, _ = get_faces_with_neighbors(disttransf)



    for key, val in faces_seg.iteritems():
        centroids=find_centroids(val, faces_dt[key], bounds[key],centroids)

    return centroids

def comparison(
        segmentation, ground_truth, segmentation_resolved

):

    edge_volume_segmentation = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmentation[:, :, z])[:, :, None] for z in
         xrange(segmentation_resolved.shape[2])],
        axis=2)
    dt_segmentation = vigra.filters.distanceTransform(edge_volume_segmentation, pixel_pitch=[1., 1., 10.],
                                                      background=True)

    centroids=compute_border_contacts(segmentation, dt_segmentation)
    return_dict={}

    for label in centroids:
        i=0
        list_gt = {}
        list_seg_res = {}

        for exit in centroids[label]:
            if i in list_gt.keys():
                list_gt[i].append(np.unique(ground_truth[exit]))
                list_seg_res[i].append(np.unique(segmentation_resolved[exit]))

            else:
                list_gt[i] = [np.unique(ground_truth[exit])]
                list_seg_res[i] = [np.unique(segmentation_resolved[exit])]
            i=i+1
        list_gt_labels=[]
        list_seg_res_labels = []
        for i in list_gt.values():
            for x in i[0]:
                list_gt_labels.append(x)
        for i in list_seg_res.values():
            for x in i[0]:
                list_seg_res_labels.append(x)
        correspondance_gt=[]
        correspondance_seg_res = []

        for x in list_gt_labels:
            a=[i for i in list_gt if x in list_gt[i][0]]
            correspondance_gt.append(a)


        for x in list_seg_res_labels:
            a=[i for i in list_seg_res if x in list_seg_res[i][0]]
            correspondance_seg_res.append(a)

        seg_res_unique=np.unique(correspondance_seg_res)
        gt_unique = np.unique(correspondance_gt)

        if type(seg_res_unique==gt_unique)== np.ndarray:

            if (seg_res_unique==gt_unique).all():
                return_dict[label]=True
            else:
                return_dict[label]=False
        else:
            if (seg_res_unique==gt_unique):
                return_dict[label]=True
            else:
                return_dict[label]=False




    return return_dict





















