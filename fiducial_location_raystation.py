'''
Sam Connolly
Version 1: June 4, 2020
Contact: saconnolly@toh.ca 

TODO:
Gets input from user as to where the fiducials should be approximately located, and wheather they
are in the liver or prostate. 
The anlysis is run for the current oppen case and primary examination. 
'''

from connect import *
import numpy as np

###################################################################
#################### define analysis functions ####################
###################################################################

def get_ct_matrix (examination, img_shape) :
    '''
    get_ct_matrix loads the current examination ct image into a numpy array.

    parameters : 
    examination (ScriptObject) : contains the current examination data from the raystation statetree
    img_shape (list) : the shape of the ct image in pixels/indecies of form [z,y,x]

    returned values :
    img3d (3d numpy array) : matrix containing the pixel values of the ct image
    '''
    # get pixel data from raystation state tree
    byte_array = examination.Series[0].ImageStack.PixelData
    # convert from .NET type <Array[Byte]> to python type <bytes>
    bytes_str = bytes(byte_array)
    # convert from 16-bit byte data into a 1D numpy array
    img1d = np.frombuffer(bytes_str, dtype=np.uint16)
    # reshape numpy array to match ct image
    img3d = np.reshape(img1d, img_shape)

    return img3d

def get_img_shape (examination) : 
    '''
    get_img_shape determines the shape of the ct image in pixels
    of the current examination.

    parameters:
    examination (ScriptObject) : contains the current examination data from the raystation statetree

    returned values : 
    img_shape (list) : the shape of the ct image in pixels/indecies of form [z,y,x]
    '''
    x_len = examination.Series[0].ImageStack.NrPixels.x
    y_len = examination.Series[0].ImageStack.NrPixels.y
    z_len = len(examination.Series[0].ImageStack.SlicePositions)
    img_shape = [z_len, y_len, x_len]
    return img_shape

def fiducial_search_area (img3d, fiducial_points_guess, img_shape, x_origin, y_origin, z_origin, fiducial_rad, pixel_spacing) :
    '''
    fiducial_search_area takes guesses for the fiducial locations and searhes for them within a certain radius through thresholding. 

    parameters :
    img3d (3d numpy array): matrix containing the pixel values of the ct image
    fiducial_points_guess (list, [z,y,x]): guesses for the fiducial locations in mm
    img_shape (list, [z,y,x]): shape of img3d
    x_origin, y_origin, z_origin (float): location of image origin relative to numpy matrix indexing
    fiducial_rad (float): radius of search for fiducials

    returnd values :
    fiducial_matrix (3d numpy array): matrix containing the threholded fiducials
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


###################################################################
#################### define gui functions #########################
###################################################################


###################################################################
#################### define program ###############################
###################################################################


def main () : 
    # get current examination loaded as primary for current case and patient
    # requires that patient is open and that there exisits a case and examination
    patient = get_current("Patient")
    case = get_current("Case")
    examination = get_current("Examination")

    # get shape of ct image for current examination
    img_shape = get_img_shape(examination)

    # get the ct as numpy array
    img3d = get_ct_matrix (examination, img_shape)


    return

if __name__ == "__main__":
    main()
