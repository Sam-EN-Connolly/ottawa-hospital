'''
Script opens and displays a one sclice of 3DCT 

Works for CT image sets of 200 

convention is always (z,y,x)

call:
$ python OpenViewCT.py ~/filepath

'''

import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
import glob
import warnings


from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import ndimage
from scipy import interpolate

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.style.use('dark_background')

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
    for fname in glob.glob('./*.dcm', recursive=False):
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
        fiducial_points_found.append(ndimage.measurements.center_of_mass(mask))

        # add fiducial to fiducial_matrix
        fiducial_matrix += mask

    #fiducial center of mass
    fiducial_cm = ndimage.measurements.center_of_mass(fiducial_matrix)

    return fiducial_matrix, fiducial_points_found, fiducial_cm

def index_to_mm (point_index, x_origin, y_origin, z_origin, pixel_spacing):
    x_mm = round(x_origin + point_index[2] * pixel_spacing[2])
    y_mm = round(y_origin + point_index[1] * pixel_spacing[1])
    z_mm = round(z_origin + point_index[0] * pixel_spacing[0])
    return [z_mm, y_mm, x_mm]

def mm_to_index (point_mm, x_origin, y_origin, z_origin, pixel_spacing):
    x_index = round((point_mm[2] - x_origin) / pixel_spacing[2])
    y_index = round((point_mm[1] - y_origin) / pixel_spacing[1])
    z_index = round((point_mm[0] - z_origin) / pixel_spacing[0])
    return [z_index, y_index, x_index]

# display functions

def plot_fiducials(img3d, fiducial_matrix, fiducial_points_found, x_origin, y_origin, z_origin, pixel_spacing) : 
    '''
    plot_fiducials plots the slices containing fiducials. 
    The inputs are the 3D matrix containing the image data (img3d), 
    the 3d numpy matrix containing the threholded fiducials (fiducial_matrix),
    and the locations of the fiducials to plot (fiducial_points_found).
    '''

    fiducial_points_mm = []
    for point in fiducial_points_found : 
        fiducial_points_mm.append(index_to_mm(point, x_origin, y_origin, z_origin, pixel_spacing))

    # plot image overlayed with fiducials
    fig, ax = plt.subplots(2, 2)   

    # fiducial 1
    fid1_y, fid1_x = np.nonzero(fiducial_matrix[int(round(fiducial_points_found[0][0])), :, :])
    ax[0,0].imshow(img3d[int(round(fiducial_points_found[0][0])), :, :], cmap='Greys_r', interpolation = 'nearest', origin = 'upper')
    ax[0,0].scatter(fid1_x, fid1_y, s=1, c='r')
    ax[0,0].set_title('Fiducial at x:'+str(round(fiducial_points_mm[0][2],1))+'mm y:'+str(round(fiducial_points_mm[0][1],1))+'mm z:'+str(round(fiducial_points_mm[0][0],1))+'mm')
    ax[0,0].xaxis.set_visible(False)
    ax[0,0].yaxis.set_visible(False)

    # fiducial 2
    fid2_y, fid2_x = np.nonzero(fiducial_matrix[int(round(fiducial_points_found[1][0])), :, :])
    ax[0,1].imshow(img3d[int(round(fiducial_points_found[1][0])), :, :], cmap='Greys_r', interpolation = 'nearest', origin = 'upper')
    ax[0,1].scatter(fid2_x, fid2_y, s=1, c='r')
    ax[0,1].set_title('Fiducial at x:'+str(round(fiducial_points_mm[1][2]))+'mm y:'+str(round(fiducial_points_mm[1][1]))+'mm z:'+str(round(fiducial_points_mm[1][0]))+'mm')
    ax[0,1].xaxis.set_visible(False)
    ax[0,1].yaxis.set_visible(False)

    # fiducial 3
    fid3_y, fid3_x = np.nonzero(fiducial_matrix[int(round(fiducial_points_found[2][0])), :, :])
    ax[1,0].imshow(img3d[int(round(fiducial_points_found[2][0])), :, :], cmap='Greys_r', interpolation = 'nearest', origin = 'upper')
    ax[1,0].scatter(fid3_x, fid3_y, s=1, c='r')
    ax[1,0].set_title('Fiducial at x:'+str(round(fiducial_points_mm[2][2]))+'mm y:'+str(round(fiducial_points_mm[2][1]))+'mm z:'+str(round(fiducial_points_mm[2][0]))+'mm')
    ax[1,0].xaxis.set_visible(False)
    ax[1,0].yaxis.set_visible(False)

    #fiducial 4
    fid4_y, fid4_x = np.nonzero(fiducial_matrix[int(round(fiducial_points_found[3][0])), :, :])
    ax[1,1].imshow(img3d[int(round(fiducial_points_found[3][0])), :, :], cmap='Greys_r', interpolation = 'nearest', origin = 'upper')
    ax[1,1].scatter(fid4_x, fid4_y, s=1, c='r')
    ax[1,1].set_title('Fiducial at x:'+str(round(fiducial_points_mm[3][2]))+'mm y:'+str(round(fiducial_points_mm[3][1]))+'mm z:'+str(round(fiducial_points_mm[3][0]))+'mm')
    ax[1,1].xaxis.set_visible(False)
    ax[1,1].yaxis.set_visible(False)

    fig.tight_layout(pad=3.0)

    plt.show()
    return

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
    structure_wanted = 'Prostate'
    list_of_ct_filenames = open_dcm_files(sys.argv[1])
    num_fiducial, fiducial_points_guess, fiducial_rad = get_fiducial_guess(structure_wanted)
    

    # analysis
    # get all the relevant data
    img3d, img_shape, x_origin, y_origin, z_origin, pixel_spacing = dcm_to_3d_arr(list_of_ct_filenames, structure_wanted)

    # find fiducials
    fiducial_matrix, fiducial_points_found, fiducial_cm = fiducial_search_area(img3d, fiducial_points_guess, img_shape,  x_origin, y_origin, z_origin, fiducial_rad, pixel_spacing)

    # plot slices with fiducial centers
    plot_fiducials(img3d, fiducial_matrix, fiducial_points_found, x_origin, y_origin, z_origin, pixel_spacing)

    return

main()

