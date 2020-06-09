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
from scipy import ndimage 

###################################################################
#################### define analysis functions ####################
###################################################################

def image_data(exam):
    '''
    image_data takes the exam object as input and returns shaped numpy array
    of pixel data
    Array dimensions:
        1 = Slices
        2 = Columns
        3 = Rows

    parameters : 
    exam (ScriptObject) : contains the current examination data from the raystation statetree

    returned values :
    img3d (3d numpy array) : matrix containing the pixel values of the ct image
    img_shape (tuple, (z,y,x)): shape of img3d in mm 
    origin (typle, (z,y,x)): location of image origin relative to numpy matrix indexing in mm
    pixel_spacing (typle, (z,y,x)): distance between the centre of pixel in all three dimentions in mm
    '''
    # determine the shape of the image
    dim_x = exam.Series[0].ImageStack.NrPixels.x
    dim_y = exam.Series[0].ImageStack.NrPixels.y
    dim_z = len(exam.Series[0].ImageStack.SlicePositions)
    img_shape = (dim_z, dim_y, dim_x)

    # determine the origin point, note sizes stored in cm in RayStation
    origin_x = -float(exam.Series[0].ImageStack.Corner.x) * 10
    origin_y = -float(exam.Series[0].ImageStack.Corner.y) * 10
    origin_z = -float(exam.Series[0].ImageStack.Corner.z) * 10
    origin = (origin_z, origin_y, origin_x)

    # determine pixel spacing, note sizes stored in cm in RayStation
    spacing_x = float(exam.Series[0].ImageStack.PixelSize.x) * 10
    spacing_y = float(exam.Series[0].ImageStack.PixelSize.y) * 10 
    spacing_z = (float(exam.Series[0].ImageStack.SlicePositions[1]) - float(exam.Series[0].ImageStack.SlicePositions[0])) * 10 
    pixel_spacing = (spacing_z, spacing_y, spacing_x)

    # determine the resclae values and 
    #rescale_intercept = exam.Series[0].ImageStack.ConversionParameters.RescaleIntercept
    #rescale_slope = exam.Series[0].ImageStack.ConversionParameters.RescaleSlope
    # determine if data is unsigner integrer or two's compliment
    pixel_representation = exam.Series[0].ImageStack.ConversionParameters.PixelRepresentation

    # extract pixel data as .NET type <Array[Byte]> and convert to numpy array
    pixel_data = exam.Series[0].ImageStack.PixelData
    pixel_bytes = bytes(pixel_data)
    pixet_float = np.frombuffer(pixel_bytes, dtype=np.uint16)
    
    # reshape array to match image shape 
    img3d = np.reshape(array, img_shape)

    # rescale HU values
    #img3d = img3d*rescale_slope+rescale_intercept

    return img3d, img_shape, origin, pixel_spacing



def fiducial_search_area (img3d, fiducial_points_guess, fiducial_rad, img_shape, origin, pixel_spacing) :
    '''
    fiducial_search_area takes guesses for the fiducial locations and searhes for them within a certain radius through thresholding. 

    parameters :
    img3d (3d numpy array): matrix containing the pixel values of the ct image
    fiducial_points_guess (list, [z,y,x]): guesses for the fiducial locations in mm
    fiducial_rad (float): radius of search for fiducials
    img_shape (tuple, (z,y,x)): shape of img3d
    origin (typle, (z,y,x)): location of image origin relative to numpy matrix indexing in mm
    pixel_spacing (typle, (z,y,x)): distance between the centre of pixel in all three dimentions in mm

    returnd values :
    fiducial_matrix (3d numpy array): matrix containing the threholded fiducials
    fiducial_points_found: the locations of the fiducials to plot
    fiducial_cm: tuple giving the location of the center of mass of all the fiducials
    '''
    fiducial_matrix = np.zeros(img_shape)
    fiducial_points_found = []
    for point in fiducial_points_guess : 
        
        # create non-zero shpere around fiducial guess in which to threshold
        center = mm_to_index (point, origin, pixel_spacing)
    
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

def get_search_radius (fiducial_points_guess, origin, pixel_spacing):
    '''
    get_search_radius takes the guess lotation for the fiducial points, and dertermines
    and appropriate radius for searching. The search radius is the half the 
    minimum distance between fiducials guess. This ensures that two fiducials are not confused.

    parameters : 
    fiducial_points_guess (list, [z,y,x]): guesses for the fiducial locations in mm
    origin (typle, (z,y,x)): location of image origin relative to numpy matrix indexing in mm
    pixel_spacing (typle, (z,y,x)): distance between the centre of pixel in all three dimentions in mm

    returned values :
    fiducial_rad (float): radius of search for fiducials
    '''
    fid_point_index = []
    distnces = []
    for val, point in enumerate(fiducial_points_guess) : 
        fid_point_index.append(mm_to_index(point, origin, pixel_spacing))
        if val > 0 : 
            squared_dist = np.sum((fid_point_index[val]-fid_point_index[val-1])**2, axis=0)
            distances.append(np.sqrt(squared_dist))

    fiducial_rad = float(min(distnces)) / 2 
    return fiducial_rad

def index_to_mm (point_index, origin, pixel_spacing):
    '''
    index_to_mm converts a point position in index values of the numpy array to positions in mm. 
    All values are rounded to nearest integer. 

    parameters : 
    point_index (list, [z_index,y_index,x_index]) : point index values to convert
    origin (typle, (z,y,x)): location of image origin relative to numpy matrix indexing in mm
    pixel_spacing (typle, (z,y,x)): distance between the centre of pixel in all three dimentions in mm

    returned values :
    point_mm (list, [z_mm,y_mm,x_mm]) : point values in mm
    '''
    x_mm = round(origin[2] + point_index[2] * pixel_spacing[2])
    y_mm = round(origin[1] + point_index[1] * pixel_spacing[1])
    z_mm = round(origin[0] + point_index[0] * pixel_spacing[0])
    return [z_mm, y_mm, x_mm]

def mm_to_index (point_mm, origin, pixel_spacing):
    '''
    index_to_mm converts a point position in mm to positions in index values.
    All values are rounded to nearest integer. 

    parameters : 
    point_mm (list, [z_mm,y_mm,x_mm]) : point values in mm
    origin (typle, (z,y,x)): location of image origin relative to numpy matrix indexing in mm
    pixel_spacing (typle, (z,y,x)): distance between the centre of pixel in all three dimentions in mm
    
    returned values :
    point_index (list, [z_index,y_index,x_index]) : point index values to convert
    '''
    x_index = round((point_mm[2] - origin[2]) / pixel_spacing[2])
    y_index = round((point_mm[1] - origin[1]) / pixel_spacing[1])
    z_index = round((point_mm[0] - origin[0]) / pixel_spacing[0])
    return [z_index, y_index, x_index]


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

    # get the ct as numpy array, and associated shape values
    img3d, img_shape, origin, pixel_spacing = image_data(exam)

    # fiducial guess for Douglas MacDonald, in mm
    fiducial_points_guess = [[-18.19, 44.85, -98.42],
                             [-18.16, 10.19, -127.00],
                             [16.38, 34.76, -65.68],
                             [28.59, 9.34, -109.03]]
    fiducial_rad = get_search_radius (fiducial_points_guess, origin, pixel_spacing)

    # get fiducials and print 
    fiducial_matrix, fiducial_points_found, fiducial_cm = fiducial_search_area(img3d, fiducial_points_guess, fiducial_rad, img_shape, origin, pixel_spacing)

    # output data to check 
    print ("The radius for fiducial search was {} \n".format(fiducial_rad))
    print ("The fiducials were found at : ")
    print ("Fiducial 1 : {}".format(fiducial_points_found[0]))
    print ("Fiducial 2 : {}".format(fiducial_points_found[1]))
    print ("Fiducial 3 : {}".format(fiducial_points_found[2]))
    print ("Fiducial 4 : {}".format(fiducial_points_found[3]))
    print ("\nThe center of mass was found to be : {}".format(fiducial_cm))
    return

if __name__ == "__main__":
    main()
