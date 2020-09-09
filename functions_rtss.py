import os
import glob
import cv2

import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 

from collections import defaultdict
from scipy.ndimage import morphology
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical


###################################################################
############################ XML files ############################
###################################################################

def convert_to_dict(contour_path):
    '''
    convert_to_dict gets all the contours from a structre set file
    and returns a dictionary of all the structure contours. 
    
    Parameters
    ----------
    contour_path : string
        Path to contour file to read

    Returns
    -------
    contours : dict
        dictionary of all contours in the file, keys as the names of the contours,
        ContourSequence as values
    contour_name : string
        name of contour manufacturer, used to identify contour types
    image_set : string
        name of image study, used to identify different image sets
    '''
    contour_file = dcm.dcmread(contour_path, force = True)
    # get contour name and image set name
    contour_name = contour_file.Manufacturer
    image_set = contour_file.StudyDescription
    ##Find the structure set objects that are to be compared
    num_to_name = dict()
    for structureSet in contour_file.StructureSetROISequence:
        num_to_name[structureSet.ROINumber] = structureSet.ROIName
    
    contours = dict()
    for contour in contour_file.ROIContourSequence:
        num = contour.ReferencedROINumber
        contours[num_to_name[num]] = contour.ContourSequence
        
    return contours, contour_name, image_set

def scale_dict(MRN, metadata_dict, contour_dict, center, scaler) : 
    '''
    scale_dict takes a contour and performs an isotropic expansion/contraction. 
    A contour is translated so that center is at [0,0,0], it's then converted to 
    spherical polar coordinates, the radius is scaled, and then it is convert back 
    to a dict in cartesian. 

    Parameters
    ----------
    contour_dict : dict
        dictionary of the contours, the keys are the slice positions, and the 
        values are the xy-point on the axial plane
    center : 1d ndarray
        the point about which to scale isotropically (np.array([z,y,x]))
    scaler : float
        amount by which to scale

    Returns
    -------
    contour_dict_scaled : dict
        dictionary of the scaled contours, the keys are the slice positions, and the 
        values are the xy-point on the axial plane
    zunique_scaled : set
        contains all z values corresponding to slices containing scaled contour values
    indexmin_scaled : 1d ndarray
        point giving the the overall minimum index of the box containing the scaled contour, for plotting
    indexmax_scaled : 1d ndarray
        point giving the the overall maximum index of the box containing the scaled contour, for plotting
    '''
    contour_dict_scaled = defaultdict(list)
    contour_dict_scaled_out = defaultdict(list)
    zunique_scaled = set()
    indexmin_scaled = np.ones(3)*np.inf
    indexmax_scaled = -np.ones(3)*np.inf

    # scale contour points, all points must be mapped to a specific plane
    for z in sorted(contour_dict.keys()) : 
        for xy_pair in contour_dict[z][0] : 
            point = np.array([z, xy_pair[1], xy_pair[0]])
            scaled_point = scale_point(point, center, scaler)
            contour_dict_scaled[scaled_point[0]].append([scaled_point[2], scaled_point[1]])
            z_scaled = scaled_point[0]
            zunique_scaled.add(z_scaled)
            # determine minimum and maximum index of 
            indexmin_scaled = np.minimum(scaled_point[::-1],indexmin_scaled)
            indexmax_scaled = np.maximum(scaled_point[::-1],indexmax_scaled)

    ROI_origin_xy = indexmin_scaled[0:2].astype(np.int32) #zero_index
    ROI_shape = np.ceil(indexmax_scaled)-np.floor(indexmin_scaled) #shape 
    ROI_shape = [int(ROI_shape[0]), int(ROI_shape[1]), len(zunique_scaled)]
    contour3d = []

    # run through contours are redraw to smooth contour
    for z in sorted(list(zunique_scaled)): 
        contour_dict_scaled[z] = [np.array(contour_dict_scaled[z])]
        # remove outliers 
        if len(contour_dict_scaled[z][0]) > 1 : 
            # determine areas and arrays of both contours and append to respective 3d matrix
            img = plot_contours(ROI_shape,ROI_origin_xy,contour_dict_scaled[z])
            contour3d.append(img)
        else : 
            contour3d.append(np.full(ROI_shape[:2], False, dtype=bool))

    # smooth contour over x direction 
    
    contour3d_x = np.transpose(contour3d, axes=(2,1,0)).astype(np.int32) #switch x and z axes

    ROI_origin_zy = np.array([0, 0], dtype=np.int32) # already plotted in indecies, no need to shift points
    ROI_shape_zy = [int(ROI_shape[2]), int(ROI_shape[1]), int(ROI_shape[0])]

    for i in range(len(contour3d_x)): 
        # find contour points from array
        contour_points = np.flip(np.nonzero(np.array(contour3d_x[i])))
        if len(contour_points[0]) > 0 : 
            contour_points = [(np.transpose(contour_points)).astype(np.int32)]
            # determine areas and arrays of both contours and append to respective 3d matrix
            img = plot_contours(ROI_shape_zy,ROI_origin_zy,contour_points)
            contour3d_x[i] = img
    contour3d = np.transpose(contour3d_x, axes=(2,1,0)) #switch x and z axes

    # smooth contour over y direction 
    contour3d_y = np.transpose(contour3d, axes=(1,0,2)).astype(np.int32) #switch y and z axes

    ROI_origin_xz = np.array([0, 0], dtype=np.int32) # already plotted in indecies, no need to shift points
    ROI_shape_xz = [int(ROI_shape[0]), int(ROI_shape[2]), int(ROI_shape[1])]

    for i in range(len(contour3d_y)): 
        # find contour points from array
        contour_points = np.flip(np.nonzero(np.array(contour3d_y[i])))
        if len(contour_points[0]) > 0 : 
            contour_points = [(np.transpose(contour_points)).astype(np.int32)]
            # determine areas and arrays of both contours and append to respective 3d matrix
            img = plot_contours(ROI_shape_xz,ROI_origin_xz,contour_points)
            contour3d_y[i] = img
    contour3d = np.transpose(contour3d_y, axes=(1,0,2)) #switch y and z axes

    # convert back to boolean surface
    contour3d_binary = np.atleast_1d(contour3d.astype(np.bool))
    # create the kernel that will be used to detect the edges of the segmentations
    conn = morphology.generate_binary_structure(contour3d_binary.ndim, 1)
    # strip edge from segmentations and subtract from origional segmentation
    # this leaves only the surface of the origional segmentation
    surface3d_binary = np.logical_xor(contour3d_binary, morphology.binary_erosion(contour3d_binary, conn))
    # convert from boolean to binary 
    surface3d = np.array(surface3d_binary, dtype = np.int32)

    # convert back to contour dict
    z_list = sorted(list(zunique_scaled))
    for i in range(len(surface3d)) : 
        z = z_list[i]
        # find contour points from array
        contour_points = np.nonzero(np.array(surface3d[i]))
        for j in range(len(contour_points[0])) : 
            point = np.array([contour_points[1][j], contour_points[0][j]]) + ROI_origin_xy
            contour_dict_scaled_out[z].append(list(point))
        contour_dict_scaled_out[z] = [np.array(contour_dict_scaled_out[z])]
    
    return contour_dict_scaled_out, zunique_scaled, indexmin_scaled, indexmax_scaled

def scale_point(point, center, scaler) : 
    '''
    scales_point scaled the radial value of a point about a given center

    Parameters
    ----------
    point : ndarray
        point to be scaled [z,y,x]
    center : 1d ndarray
        the point about which to scale isotropically (np.array([z,y,x]))
    scaler : float
        amount by which to scale

    Returns
    -------
    scaled_point : list
        the scaled point, with and integer z value [z,y,x]
    '''
    centered_point = np.array(point) - center # shift point so that centroid is located at [0,0,0], point and center must be ndarrays
    polar_point = cartesian_to_spherical(centered_point[2], centered_point[1], centered_point[0])
    scaled_polar_point = [scaler * polar_point[0], polar_point[1], polar_point[2]] # scale radius relative to centroid
    scaled_centered_point = np.array(spherical_to_cartesian(scaled_polar_point[0], scaled_polar_point[1], scaled_polar_point[2]))
    scaled_point = np.add(scaled_centered_point[::-1], center) # reverse point to be [z,y,x] and revert back to origional location
    return [round(scaled_point[0]), scaled_point[1], scaled_point[2]]

def get_contour_points_dict(contours, structures_wanted):
    '''
    get_contour_points_dict takes a dictionary of all the contours in a structure file
    and builds a new dictionary of the desired structure. 

    Parameters
    ----------
    contours : dict
        dictionary of all contours in the file, keys as the names of the contours,
        ContourSequence as values
    structures_wanted

    Returns
    -------
    contour_dict : dict
        dictionary of the contours, the keys are the slice positions, and the 
        values are the xy-point on the axial plane
    zunique : set
        contains all z values corresponding to slices containing contour values
    indexmin : 1d ndarray
        point giving the the overall minimum index of the box containing the contour, for plotting
    indexmax : 1d ndarray
        point giving the the overall maximum index of the box containing the contour, for plotting
    '''
    # check if structure in structure sets
    structure_found = False

    for structure in structures_wanted : 
        if structure in contours.keys() and not structure_found : 
            structure_wanted = structure
            structure_found = True
    if not structure_found : 
        print("Missing structure set")
        return None, None, None, None
    
    contour_dict = defaultdict(list)
    zunique = set()
    indexmin = np.ones(3)*np.inf
    indexmax = -np.ones(3)*np.inf

    # get contour points and reshape inton ndarray for plotting
    c = contours[structure_wanted]
    for contourSlice in c:
        contourData = np.array(contourSlice.ContourData)
        contourData = contourData.reshape((len(contourData)//3,3))
        z = round(contourData[2,2])
        zunique.add(z)
        contour_dict[z].append(contourData[:,0:2])
        # get minimum and maximum indecies to plot within 
        indexmin = np.minimum(contourData.min(axis = 0),indexmin)
        indexmax = np.maximum(contourData.max(axis = 0),indexmax) 

    return contour_dict, zunique, indexmin, indexmax

def get_volume_and_center_ROI(contour_dict, zunique, indexmin, indexmax) :
    '''
    get_volume_and_center_ROI takes the contour dict and it's size indicators, the 
    difference slice values, and the coners of the box containing the contour
    to find the volume and the center of the ROI.  
    Note, the volume is overestimated in that every voxel crossed by the contour
    is considered part of the contour.   

    Parameters
    ----------
    contour_dict : dict
        dictionary of the contours, the keys are the slice positions, and the 
        values are the xy-point on the axial plane
    zunique : set
        contains all z values corresponding to slices containing contour values
    indexmin : 1d ndarray
        point giving the the overall minimum index of the box containing the contour, for plotting
    indexmax : 1d ndarray
        point giving the the overall maximum index of the box containing the contour, for plotting

    Returns
    -------
    volume : float
        the volume of the contour.
    center_of_mass : tuple 
        the center of mass of the contour.
    '''

    margin = 0    
    shape =(np.ceil(indexmax)-np.floor(indexmin)+2*margin) # determine size of smallest box around contour, with any margin padding
    shape = [int(i) for i in shape] # convert box indecies to integers
    zero_index = indexmin[0:2].astype(np.int32)-margin # determine corner of contour, or zero point
    
    slices = list()
    volume = 0

    z_total = 0
    y_total = 0
    x_total = 0
    num_points = 0

    for z in sorted(list(zunique)):

        # determine areas and arrays of both contours and append to respective 3d matrix
        img = plot_contours(shape,zero_index,contour_dict[z])
        slices.append(img)
        volume += sum(sum(img))

    center_of_mass = ndimage.measurements.center_of_mass(np.array(slices)) 
    center_of_mass = center_of_mass + np.flip(indexmin)
    
    return volume, center_of_mass

def calc_similarity_metrics(contour_dict_1, zunique_1, indexmin_1, indexmax_1, contour_dict_2, zunique_2, indexmin_2, indexmax_2):
    '''
    get_contour_points_dict takes a dictionary of all the contours in a structure file
    and builds a new dictionary of the desired structure. 

    Parameters
    ----------
    contour_dict_1 : dict
        dictionary of the contours, the keys are the slice positions, and the 
        values are the xy-point on the axial plane, for first image
    zunique_1 : set
        contains all z values corresponding to slices containing contour values, for first image
    indexmin_1 : 1d ndarray
        point giving the the overall minimum index of the box containing the contour, for plotting, for first image
    indexmax_1 : 1d ndarray
        point giving the the overall maximum index of the box containing the contour, for plotting, for first image
    contour_dict_2 : dict
        dictionary of the contours, the keys are the slice positions, and the 
        values are the xy-point on the axial plane, for second image
    zunique_2 : set
        contains all z values corresponding to slices containing contour values, for second image
    indexmin_2 : 1d ndarray
        point giving the the overall minimum index of the box containing the contour, for plotting, for second image
    indexmax_2 : 1d ndarray
        point giving the the overall maximum index of the box containing the contour, for plotting, for second image

    Returns
    -------
    msd : float
        means surface distance
    rms : float
        residual mean-square error
    hd : float 
        hausdorff distance
    dsc : flaot 
        dice coefficient
    '''
    margin = 10
    contours = [contour_dict_1, contour_dict_2]
    
    indexmin = np.minimum(indexmin_1, indexmin_2)
    indexmax = np.maximum(indexmax_1, indexmax_2)
    zunique = zunique_1.union(zunique_2)   
    
    shape =(np.ceil(indexmax)-np.floor(indexmin)+2*margin) # determine size of smalles box around contour, with any margin padding
    shape = [int(i) for i in shape] # convert box indecies to integers
    zero_index = indexmin[0:2].astype(np.int32)-margin # determine corner of contour, or zero point
    
    set_1 = 0
    set_2 = 0 
    set_overlap = 0
    slices1 = list()
    slices2 = list()

    for z in sorted(list(zunique)):

        # determine areas and arrays of both contours and append to respective 3d matrix
        img1 = plot_contours(shape,zero_index,contours[0][z])
        img2 = plot_contours(shape,zero_index,contours[1][z])
        slices1.append(img1)
        slices2.append(img2)
    
        # calculate set cardinality and overlap for dice coefficient
        set_1 += sum(sum(img1))
        set_2 += sum(sum(img2))
        set_overlap += sum(sum(np.logical_and(img1,img2)))
    
    # convert slices list to ndarray
    surface_1 = np.array(slices1)
    surface_2 = np.array(slices2)

    # calculate surface distance 
    surface_distance = find_surface_distance(surface_1, surface_2)
    
    # calculate mean surface distance, residual mean surface distance, and hausdorff distance
    msd = surface_distance.mean()
    rms = np.sqrt((surface_distance**2).mean())
    hd  = surface_distance.max()

    # calculate dice coefficient
    dsc = 2*set_overlap/(set_1+set_2)
    
    return round(msd,2), round(rms,2), round(hd,2), round(dsc,2)

def plot_contours(shape, zero_index, contour_list):
    '''
    Given shape (the size of the array), zero_index (the zero position in mm), and 
    contour list (the list of contour points to plot), plot the contours in 2D
    and filled area of contour. Keep only values greater than zero as True in boolean array.

    This method deems every pixel crossed by the contour to be a part of the contour. 
    Resolution of the contour is at best the size of the pixel. 

    Parameters
    ----------
    shape : list (int) 
        shape of image to plot as [x,y,z] or [x,y]. This is usually the minimum bounding box
        of the contour 
    zero_index : list (int)
        the corner to zero index or origin as [x,y]. This determines the amount to translate
        the points to be plotted in a smaller box. If the points are already in indecies, and 
        not mm, this can be set to [0.0]
    contour_list : list
        a list of the contour points form a contour dict

    Returns
    -------
    img_plot : array (boolean)
        the 2D plotted image, retured as a boolean array, where all points within the contour
        are True, and all points outside the contour are False
    '''
    contours_plot = [b.astype(np.int32) - zero_index for b in contour_list]
    img_plot = np.zeros((shape[1],shape[0]))
    if len(contours_plot)>0:
        cv2.fillPoly(img_plot, pts=contours_plot, color=(1,1,1))
        #cv2.imshow('image',img_plot)
        #cv2.waitKey(0)

    return img_plot > 0    

def find_surface_distance(input1, input2, sampling=1, connectivity=1):
    '''
    find_surface_distance calculates the surface distance between two surfaces.
    It can be used to easily find the Mean Surface Distance (MSD), 
    Residual Mean-Square Error (RMS) and the Hausdorff Distance (HD).
        msd = surface_distance.mean()
        rms = np.sqrt((surface_distance**2).mean())
        hd  = surface_distance.max()
    Surface distance is an estimate of the error between outer surfaces, S and S'
    The distance between a point p on surface S and the surface S' is given by the 
    minimum of the Euclidean norm:
        d(p,S') = min{p' contained in S'} (||p-p'||)
    Doing this for all pixels in the surface gives the total surface 
    distance between S and S' : d(S,S')

    Parameters
    ----------
    input1 : ndarray (3d)
        The set (segmentation) that will be compared to the ground truth
    input2 : ndarray (3d)
        The set (segmentation) that is considered the ground truth
    sampling : list/tuple (3d)
        The pixel resolution or pixel size
    connectivity : int
        Determines conectivity of surface

    Returns
    -------
    sds : ndarray (1d)
        symetric suraface distances between surfaces

    This function is origionally taken from : https://mlnotebook.github.io/post/surface-distance-function/
    '''
    # check size and array and convert to boolean 
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    # create the kernel that will be used to detect the edges of the segmentations
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    # strip edge from segmentations and subtract from origional segmentation
    # this leaves only the surface of the origional segmentation
    S = np.logical_xor(input_1, morphology.binary_erosion(input_1, conn))
    Sprime = np.logical_xor(input_2, morphology.binary_erosion(input_2, conn))

    # create feild of euclidean distances from surface
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    # create and flatten surface distance array 
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds
    
if __name__ == "__main__":
    pass