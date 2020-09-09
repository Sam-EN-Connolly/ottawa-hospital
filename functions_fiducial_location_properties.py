import numpy as np

def average_fid_distance(fiducial_points, center):
    '''
    average_fid_distance determines the average distance of the
    fiducials form their collective center of mass.

    Parameters
    ----------
    fiducial_points: ndarray (2d)
        The locations of the fiducials
    center: tuple
        The location of the center about which to find fiducial distances 

    Returns
    -------
    mean_distances : float
        The mean distance of the fiducials from the fiducial center of mass

    '''
    if len(fiducial_points) < 2 : 
        return 0
    distances = []
    # for each fiducial point, determine it's distance to the center
    for point in fiducial_points : 
        squared_dist = np.sum((np.array(point)-np.array(center))**2, axis=0)
        distances.append(np.sqrt(squared_dist))

    return np.mean(distances)

def change_fid_positions(fiducial_points_1,fiducial_points_2) : 
    '''
    average_fid_distance determines the average distance from 
    each fiducial in image 1, to it's corresponding fiducial in image 2. 
    The two images must have the same number of fiducials, and 
    the order of the fiducials must be the same. 

    Parameters
    ----------
    fiducial_points_1: ndarray (2d)
        The locations of the fiducials in one image
    fiducial_points_1: ndarray (2d)
        The locations of the fiducials in a second image

    Returns
    -------
    position_change : ndarray
        The difference in location of each fiducial

    '''
    if len(fiducial_points_1) != len(fiducial_points_2) : 
        print('Images must have number of fiducials')
        return None

    fid_arr_1 = np.array(fiducial_points_1)
    fid_arr_2 = np.array(fiducial_points_2)
    position_change = []
    for point_num in range(len(fid_arr_1)) : 
        squared_dist = np.sum((np.array(fid_arr_1[point_num])-np.array(fid_arr_2[point_num]))**2, axis=0)
        position_change.append(np.sqrt(squared_dist))

    return np.array(position_change)

def average_interfiducial_distance (fiducial_points):
    '''
    average_interfiducial_distance determines the average distance 
    between the fiducials.

    Parameters
    ----------
    fiducial_points: ndarray (2d)
        The locations of the fiducials

    Returns
    -------
    mean_distances : float
        The mean distance of between fiducials 

    '''
    if len(fiducial_points) < 2 : 
        return 0
    distances = []
    for i in range(len(fiducial_points)) : 
        for j in range(i+1, len(fiducial_points)) :
            fid1 = fiducial_points[i]
            fid2 = fiducial_points[j]
            squared_dist = np.sum((np.array(fid1)-np.array(fid2))**2, axis=0)
            distances.append(np.sqrt(squared_dist))

    return np.mean(distances)

def get_iso_scale_factor(fid_distance_1, fid_distance_2) : 
    '''
    get_iso_scale_factor determins the scaling factor for an iostropic 
    expansion/contraction of the ROI. It is the scaling facor such that the 
    average distance of fid_distances_2 matches the average distance of fid_distances_1. 
    Note, there must be the same number of fiducials in both images. 

    Where C is the scaling factor, N is the number of fiducials, 
    the distances to scaled are marked prime, equate the average distances : 

        (r1 + r2 + ... + rN) / N = (Cr1' + Cr2' + ... + CrN') / N
        C = (r1 + r2 + ... + rN) / (r1' + r2' + ... + rN')
        C = average_distance_1 / average_distance_2
    
    Parameters
    ----------
    fid_distance_1: float 
        The average distance of the fiducials to the center of the ROI. Ususally the TPCT image
    fid_distance_2: float 
        The average distance of the fiducials to the center of the ROI in the image to be scaled. 
        Ususally the Fused image

    Returns
    -------
    scaling_factor : float
        The scaling factor by which to scale fid_distances_2 for an isotropic expansion
    '''
    scaling_factor = fid_distance_1 / fid_distance_2
    return scaling_factor