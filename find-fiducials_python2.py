'''
Script opens and displays a one sclice of 3DCT 

Works for CT image sets of 200 

convention is always (z,y,x)

call:
$ python OpenViewCT.py ~/filepath

'''

import pydicom as dcm
import numpy as np
import sys
import os 
import glob
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# define global variables

def open_dcm_files(dir_path) : 
    '''
    open_decm_files opens a directory of .dcm files and returns a list of the read files. 
    Parameter dir_path is a string of the path to the directory with the /dcm files.
    '''
    # load the DICOM files
    files = []
    os.chdir(dir_path)                                   # move to directory with DICOM files
    print('glob: {}'.format(dir_path))
    for fname in glob.glob('./*.dcm'):
        #print("loading: {}".format(fname))                 # see loaded files printed
        files.append(dcm.dcmread(fname))
    print("file count: {}".format(len(files)))

    return files

def dcm_to_3d_arr (dcm_files, structure_wanted) : 
    '''
    dcm_to_3d_arr takes a list of .dcm files and converts them to a 3D numpy array. 
    The functions return the 3D array, the shape of the array, 
    and the aspect ratios of the three dimentions based on the pixel 
    spacing and slice thickness given in the files. 
    Only dcm files with the attribute SliceLocation are used. 
    Parameter is dcm_files, which is a list of open .dcm files.
    '''

    # skip files with no SliceLocation (eg scout views)
    ct_slices = []
    skipcount = 0
    for f in dcm_files:
        if hasattr(f, 'SliceLocation'):
            ct_slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    ct_slices = sorted(ct_slices, key=lambda s: s.SliceLocation)     

    # create 3D array
    img_shape = [len(ct_slices)] + (list(ct_slices[0].pixel_array.shape))
    img3d = np.zeros(img_shape)

    #Get positional info for later plotting of structure contours
    x_origin = float(ct_slices[0].ImagePositionPatient[0])
    y_origin = float(ct_slices[0].ImagePositionPatient[1])
    z_origin = float(ct_slices[0].ImagePositionPatient[2])
    pixel_spacing = [float(ct_slices[0].SliceThickness), float(ct_slices[0].PixelSpacing[0]), float(ct_slices[0].PixelSpacing[0])]  #mm pixel spacing [z,y,x]
        
    # fill 3D array with the images from the files
    for i, s in enumerate(ct_slices):
        img2d = s.pixel_array
        img3d[i,:, :] = img2d   


    return img3d, img_shape, x_origin, y_origin, z_origin, pixel_spacing

def fiducial_search_area (img3d, fiducial_points_guess, img_shape, x_origin, y_origin, z_origin, fiducial_rad, pixel_spacing):
    '''
    fiducial_search_area takes guesses for the fiducial locations and searhes for them within a certain radius through thresholding. 
    the function takes:
    img3d: 3d numpy matrix containing the ct image
    fiducial_points_guess: guesses for the fiducial locations, list of points, each a list of x, y, and z location
    img_shape: shape of img3d, [z, y, x]
    x_origin, y_origin, z_origin: location of image origin relative to numpy matrix indexing
    fiducial_rad: floatgiving radius of search for fiducials

    function returnd:
    fiducial_matrix: the 3d numpy matrix containing the threholded fiducials
    fiducial_points_found: the locations of the fiducials to plot
    fiducial_cm: tuple giving the location of the center of mass of all the fiducials
    '''
    fiducial_matrix = np.zeros(img_shape)
    fiducial_points_found = []
    for point in fiducial_points_guess : 
        
        # create non-zero shpere around fiducial guess in which to threshold
        center = mm_to_index (point, x_origin, y_origin, z_origin, pixel_spacing)
    
        distance = np.linalg.norm(np.subtract(np.indices(img_shape).T,np.asarray(center)), axis=len(center)).T
        mask = np.ones(img_shape) * (distance<=fiducial_rad) * img3d

        # threshold to find fiducial
        thresh = 2000
        mask[mask<=thresh] = 0

        # add fiducial center of mass
        fiducial_points_found.append(center_of_mass(mask))

        # add fiducial to fiducial_matrix
        fiducial_matrix += mask

    #fiducial center of mass
    fiducial_cm = center_of_mass(fiducial_matrix)

    return fiducial_matrix, fiducial_points_found, fiducial_cm

def index_to_mm (point_index, x_origin, y_origin, z_origin, pixel_spacing, rounding=False):
    x_mm = (x_origin + point_index[2] * pixel_spacing[2])
    y_mm = (y_origin + point_index[1] * pixel_spacing[1])
    z_mm = (z_origin + point_index[0] * pixel_spacing[0])

    if rounding : 
        return [round(z_mm), round(y_mm), round(x_mm)]

    return [z_mm, y_mm, x_mm]

def mm_to_index (point_mm, x_origin, y_origin, z_origin, pixel_spacing, rounding=True):
    x_index = ((point_mm[2] - x_origin) / pixel_spacing[2])
    y_index = ((point_mm[1] - y_origin) / pixel_spacing[1])
    z_index = ((point_mm[0] - z_origin) / pixel_spacing[0])

    if rounding : 
        return [round(z_index), round(y_index), round(x_index)]

    return [z_index, y_index, x_index]

def center_of_mass(input):
    """
    Calculate the center of mass of the values of an array at labels.
    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.
    """
    normalizer = np.sum(input)
    grids = np.ogrid[[slice(0, i) for i in input.shape]]

    results = [np.sum(input * grids[dir].astype(float)) / normalizer
               for dir in range(input.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]

def get_fiducial_guess(structure_wanted) :
    '''
    Prompts user for number of fiducials and their x, y, and z locations in mm coordinates.
    Points given as [z,y,x]
    '''
    global num_fiducial, fiducial_points_guess, fiducial_rad
    num_fiducial = 4
    if structure_wanted == 'Liver':
        fiducial_points_guess = [[-18.19, 44.85, -98.42],
                                [-18.16, 10.19, -127.00],
                                [16.38, 34.76, -65.68],
                                [28.59, 9.34, -109.03]]
        fiducial_rad = 5                                            # radius around which to search for fiducials
    if structure_wanted == 'Prostate':
        fiducial_points_guess = [[-13.75, 111.25, 6.25],
                                [-25.82, 137.76, 10.88],
                                [-25.93, 134.31, -28.22],
                                [-20.40, 140.20, -30.40]]
        fiducial_rad = 5                                            # smaller radius as fiducials are closer together 
    return num_fiducial, fiducial_points_guess, fiducial_rad

def main() : 

    # define initial variables
    structure_wanted = 'Liver'
    list_of_ct_filenames = open_dcm_files(sys.argv[1])
    num_fiducial, fiducial_points_guess, fiducial_rad = get_fiducial_guess(structure_wanted)
    

    # analysis
    # get all the relevant data
    img3d, img_shape, x_origin, y_origin, z_origin, pixel_spacing = dcm_to_3d_arr(list_of_ct_filenames, structure_wanted)

    # find fiducials
    fiducial_matrix, fiducial_points_found, fiducial_cm = fiducial_search_area(img3d, fiducial_points_guess, img_shape,  x_origin, y_origin, z_origin, fiducial_rad, pixel_spacing)

    fiducial_points_mm = []
    for point in fiducial_points_found : 
        fiducial_points_mm.append(index_to_mm (point, x_origin, y_origin, z_origin, pixel_spacing))

    # output data to check 
    print ("The radius for fiducial search was {} \n".format(fiducial_rad))
    print ("The fiducials were found at : ")
    print ("Fiducial 1 : {}".format(fiducial_points_mm[0]))
    print ("Fiducial 2 : {}".format(fiducial_points_mm[1]))
    print ("Fiducial 3 : {}".format(fiducial_points_mm[2]))
    print ("Fiducial 4 : {}".format(fiducial_points_mm[3]))
    print ("\nThe center of mass was found to be : {}".format(index_to_mm (fiducial_cm, x_origin, y_origin, z_origin, pixel_spacing)))
    
    return

main()

