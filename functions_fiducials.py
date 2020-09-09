import os 
import glob
import warnings

import numpy as np
from scipy import ndimage
import pydicom as dcm

from functions_rtss import *
warnings.filterwarnings("ignore", category=RuntimeWarning)


###################################################################
#################### define analysis functions ####################
###################################################################

def open_dcm_files(dir_path) : 
    '''
    open_decm_files opens a directory of .dcm files and returns a list of the read files
    and the rtss file if it exists 
    
    Parameters
    ----------
    dir_path : string
        Path to directory containing DICOM files

    Returns
    -------
    dcm_files : list
        A list of the dcm files from which to get image data
    rts_filename : string
        Filename of rtss file containing structure sets
    '''
    # load the DICOM files
    files = []
    print('glob: {}'.format(dir_path))
    for fname in glob.glob(os.path.join(dir_path,'*.dcm'), recursive=None):
        # check that file is not a structure file
        if 'rtss' or 'RTS' not in fname :          
            files.append(dcm.dcmread(fname))
    print("file count: {}".format(len(files)))

    return files

def dcm_to_3d_arr (dcm_files) : 
    '''
    dcm_to_3d_arr takes a list of .dcm files as input and returns shaped numpy array
    of pixel data
    Array dimensions:
        1 = Slices
        2 = Columns
        3 = Rows

    Parameters
    ----------
    dcm_files : list
        A list of the dcm files from which to get image data

    Returns
    -------
    img3d : ndarray (3d)
        Matrix containing the pixel values of the ct image
    img_shape : tuple (z,y,x)
        Shape of img3d in mm 
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm
    '''

    # skip files with no SliceLocation (eg scout views)
    ct_slices = []
    skipcount = 0
    for f in dcm_files:
        # check if file has sclice location, and skip if not
        # This skips structure files tha tmight be in the folder
        if hasattr(f, 'SliceLocation'):
            ct_slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order, sorted based on slice location
    ct_slices = sorted(ct_slices, key=lambda s: s.SliceLocation)

    # create 3D array
    img_shape = [len(ct_slices)] + (list(ct_slices[0].pixel_array.shape))
    img3d = np.zeros(img_shape)

    # get positional info for later plotting of structure contours
    x_origin = float(ct_slices[0].ImagePositionPatient[0])
    y_origin = float(ct_slices[0].ImagePositionPatient[1])
    z_origin = float(ct_slices[0].ImagePositionPatient[2])
    origin = (z_origin, y_origin, x_origin)
    pixel_spacing = (float(ct_slices[0].SliceThickness), float(ct_slices[0].PixelSpacing[0]), float(ct_slices[0].PixelSpacing[0]))  #mm pixel spacing (z,y,x)

    # fill 3D array with the images from the files
    for i, s in enumerate(ct_slices):
        img2d = s.pixel_array
        img3d[i,:, :] = img2d

    return img3d, img_shape, origin, pixel_spacing

def fiducial_search_area (img3d, fiducial_points_guess, img_shape, origin, fiducial_rad, pixel_spacing):
    '''
    fiducial_search_area takes guesses for the fiducial locations and searhes for 
    them within a certain radius through thresholding. 

    Parameters
    ----------
    img3d : ndarray (3d)
        Matrix containing the pixel values of the ct image
    fiducial_points_guess : list [z,y,x]
        Guesses for the fiducial locations in mm
    fiducial_rad : float
        Radius of search for fiducials
    img_shape : tuple (z,y,x)
        Shape of img3d in mm 
    origin : tuple (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    fiducial_matrix : ndarray (3d) 
        Matrix containing the threholded fiducials
    fiducial_points_found: ndarray (2d) [[z,y,x], ...]
        The locations of the fiducials in mm
    fiducial_cm: tuple (z,y,x)
        The location of the center of mass of all the fiducials in mm
    '''
    fiducial_matrix = np.zeros(img_shape)
    fiducial_points_found = []

    for point in fiducial_points_guess : 
        
        # create non-zero shpere around fiducial guess in which to threshold
        center = mm_to_index(point, origin, pixel_spacing)
        distance = np.linalg.norm(np.subtract(np.indices(img_shape).T,np.asarray(center)), axis=len(center)).T
        mask = np.ones(img_shape) * (distance<=fiducial_rad) * img3d

        # threshold to find fiducial
        thresh = np.amax(img3d) / 2
        mask[mask<=thresh] = 0

        # add fiducial center of mass
        fid_point_found = ndimage.measurements.center_of_mass(mask)
        fid_point_found_mm = index_to_mm(fid_point_found, origin, pixel_spacing)
        fiducial_points_found.append(fid_point_found_mm.tolist())

        # add fiducial to fiducial_matrix
        fiducial_matrix += mask

    # fiducial center of mass
    fiducial_cm = ndimage.measurements.center_of_mass(fiducial_matrix)
    fiducial_cm = index_to_mm(np.flip(fiducial_cm), origin, pixel_spacing)

    # convert lists to arrays
    fiducial_points_found = np.array(fiducial_points_found)

    return fiducial_matrix, fiducial_points_found, fiducial_cm

def get_search_radius (fiducial_points_guess, origin, pixel_spacing):
    '''
    get_search_radius takes the guess lotation for the fiducial points, and dertermines
    and appropriate radius for searching. The search radius is the half the 
    minimum distance between fiducials guess. This ensures that two fiducials are not confused.

    Parameters
    ----------
    fiducial_points_guess : list (2d) [[z,y,x], ... ]
        Guesses for the fiducial locations in mm
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    fiducial_rad : float
        Radius of search for fiducials
    '''
    if len(fiducial_points_guess) < 2 : 
        return 10
    fid_point_index = []
    distances = []
    # determine distances between fiducials
    for val, point in enumerate(fiducial_points_guess) : 
        fid_point_index.append(np.array(mm_to_index(point, origin, pixel_spacing)))
        if val > 0 : 
            squared_dist = np.sum((fid_point_index[val]-fid_point_index[val-1])**2, axis=0)
            distances.append(np.sqrt(squared_dist))

    # determine minimum distance between fiducials
    fiducial_rad = min(float(min(distances))/2, 10.0)

    return fiducial_rad


###################################################################
#################### math functions ###############################
###################################################################

def index_to_mm(point_index, origin, pixel_spacing, rounding=True):
    '''
    index_to_mm converts a point position in index values of the numpy array to positions in mm. 
    All values are rounded to nearest integer. 

    Parameters
    ----------
    point_index : ndarray [z_index,y_index,x_index]
        Point index values to convert
    origin : tuple (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    point_mm : ndarray [z_mm,y_mm,x_mm]
        Point values in mm
    '''
    x_mm = (origin[2] + point_index[2] * pixel_spacing[2])
    y_mm = (origin[1] + point_index[1] * pixel_spacing[1])
    z_mm = (origin[0] + point_index[0] * pixel_spacing[0])

    # round to two decimal places, since mest resolution is 1mm
    if rounding : 
        return np.array([round(z_mm,2), round(y_mm,2), round(x_mm,2)])

    return np.array([z_mm, y_mm, x_mm])

def mm_to_index (point_mm, origin, pixel_spacing, rounding=True):
    '''
    mm_to_index converts a point position in mm to positions in index values.
    All values are rounded to nearest integer. 

    Parameters
    ----------
    point_mm : ndarray [z_mm,y_mm,x_mm]
        Point values in mm to convert
    origin : tuple (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    point_index : ndarray [z_index,y_index,x_index]
        Point index values 
    '''
    x_index = ((point_mm[2] - origin[2]) / pixel_spacing[2])
    y_index = ((point_mm[1] - origin[1]) / pixel_spacing[1])
    z_index = ((point_mm[0] - origin[0]) / pixel_spacing[0])

    # round to nearest index, so that these values can be used to access index points in arrays
    # returned values are then ints
    if rounding : 
        return np.array([round(z_index), round(y_index), round(x_index)])

    return np.array([z_index, y_index, x_index])


