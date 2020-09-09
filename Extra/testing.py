# -*- coding: utf-8 -*-
import pydicom as dcm
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
import os
import glob
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

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

def scale_dict(contour_dict, center, scaler) : 
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
    zunique_scaled
    indexmin_scaled
    indexmax_scaled
    '''
    contour_dict_scaled = defaultdict(list)
    zunique_scaled = set()
    indexmin_scaled = np.ones(3)*np.inf
    indexmax_scaled = -np.ones(3)*np.inf
    for z in contour_dict.keys() : 
        for xy_pair in contour_dict[z][0] : 
            point = np.array([z, xy_pair[1], xy_pair[0]])
            scaled_point = scale_point(point, center, scaler)
            contour_dict_scaled[scaled_point[0]].append([scaled_point[2], scaled_point[1]])
            z = scaled_point[0]
            zunique_scaled.add(z)
            indexmin_scaled = np.minimum(scaled_point[::-1],indexmin_scaled)
            indexmax_scaled = np.maximum(scaled_point[::-1],indexmax_scaled)
            
    for z in contour_dict_scaled.keys() : 
        contour_dict_scaled[z] = [np.array(contour_dict_scaled[z])]
        
    return contour_dict_scaled, zunique_scaled, indexmin_scaled, indexmax_scaled

def scale_point(point, center, scaler) : 
    '''
    scales_point scaled the radial value of a point about a given center

    scale_dict takes a contour and performs an isotropic expansion/contraction. 
    A contour is translated so that center is at [0,0,0], it's then converted to 
    spherical polar coordinates, the radius is scaled, and then it is convert back 
    to a dict in cartesian. 

    Parameters
    ----------
    point : list
        point to be scaled [z,y,x]
    center : 1d ndarray
        the point about which to scale isotropically (np.array([z,y,x]))
    scaler : float
        amount by which to scale

    Returns
    -------
    scaled_point : list
        the scaled point, with and integer z value [z,y,x]
    zunique_scaled : set
        contains all z values corresponding to slices containing scaled contour values
    indexmin_scaled : 1d ndarray
        point giving the the overall minimum index of the box containing the scaled contour, for plotting
    indexmax_scaled : 1d ndarray
        point giving the the overall maximum index of the box containing the scaled contour, for plotting
    '''
    centered_point = point - center # shift point so that centroid is located at [0,0,0], point and center must be ndarrays
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


def plot_contours(x, zeroIdx, contour_list):
    '''
    Given x, the size of the array, zeroIdx the zero position in mm, and 
    contour list, the list of contour points to plot, plot the contours in 2D'''
    contours_plot = [b.astype(np.int32) - zeroIdx for b in contour_list]
    img_plot = np.zeros((x[1],x[0]))
    if len(contours_plot)>0:
        cv2.fillPoly(img_plot, pts =contours_plot, color=(1,1,1))
        
    return img_plot > 0    


def testing() : 
    rts_folder = r'/Volumes/External Drive/Testing/CT_files/09115957/RTS/'
    structure_wanted = 'Prostate'
    structures_wanted = [structure_wanted]
    rts_files = glob.glob(os.path.join(rts_folder,'*.dcm'), recursive=None)
