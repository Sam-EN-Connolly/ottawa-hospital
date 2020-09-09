'''
Sam Connolly
Version 1: June 4, 2020
Contact: saconnolly@toh.ca 

This program reads the current open examination image data and finds the location of the fiducials.
Currently the fidicuial search locations are hard-coded in for testing purposes. 
It requires that a patien, case, and examination are open. 
'''

import sys
import numpy as np 
from connect import *

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

    Parameters
    ----------
    exam : pyScriptObject
        The current examination data from the raystation statetree

    Returns
    -------
    img3d : ndarray (3d)
        Matrix containing the pixel values of the ct image
    img_shape : tuple (z,y,x)
        Shape of img3d in cm 
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm
    '''
    # determine the shape of the image
    dim_x = exam.Series[0].ImageStack.NrPixels.x
    dim_y = exam.Series[0].ImageStack.NrPixels.y
    dim_z = len(exam.Series[0].ImageStack.SlicePositions)
    img_shape = (dim_z, dim_y, dim_x)

    # determine the origin point, note sizes stored in cm in RayStation
    origin_x = float(exam.Series[0].ImageStack.Corner.x)
    origin_y = float(exam.Series[0].ImageStack.Corner.y)
    origin_z = float(exam.Series[0].ImageStack.Corner.z)
    origin = (origin_z, origin_y, origin_x)

    # determine pixel spacing, note sizes stored in cm in RayStation
    spacing_x = float(exam.Series[0].ImageStack.PixelSize.x)
    spacing_y = float(exam.Series[0].ImageStack.PixelSize.y)
    spacing_z = float(exam.Series[0].ImageStack.SlicePositions[1]) - float(exam.Series[0].ImageStack.SlicePositions[0])
    pixel_spacing = (spacing_z, spacing_y, spacing_x)

    # determine the resclae values to turn pixel values to HU values
    #rescale_intercept = exam.Series[0].ImageStack.ConversionParameters.RescaleIntercept
    #rescale_slope = exam.Series[0].ImageStack.ConversionParameters.RescaleSlope
    
    # determine if data type of pixel, unsigned integer or two's compliment
    pixel_representation = exam.Series[0].ImageStack.ConversionParameters.PixelRepresentation


    pixel_data = exam.Series[0].ImageStack.PixelData

    length = len(pixel_data)

    # convert from 16-bit to integers
    evens = np.arange(0, length, 2, dtype=np.int)
    odds = np.arange(1, length, 2, dtype=np.int)
    
    if pixel_representation == 0:
        pixel_arr = (pixel_data[evens] + pixel_data[odds] * 265)
    else:
        print("Pixel representation is not unsigned integer, exiting script")
        sys.exit()
    
    # reshape array to match image shape 
    img3d = np.reshape(pixel_arr, img_shape)
    #img3d = img3d.astype(np.float64)

    # rescale to HU values
    #img3d = img3d*rescale_slope+rescale_intercept

    return img3d, img_shape, origin, pixel_spacing



def fiducial_search_area (exam, img3d, fiducial_points_guess, fiducial_rad, img_shape, origin, pixel_spacing) :
    '''
    fiducial_search_area takes guesses for the fiducial locations and searhes for them within a certain radius through thresholding. 

    Parameters
    ----------
    exam : pyScriptObject
        The current examination data from the raystation statetree
    img3d : ndarray (3d)
        Matrix containing the pixel values of the ct image
    fiducial_points_guess : list [z,y,x]
        Guesses for the fiducial locations in cm
    fiducial_rad : float
        Radius of search for fiducials
    img_shape : tuple (z,y,x)
        Shape of img3d in cm 
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm

    Returns
    -------
    fiducial_matrix : ndarray (3d) 
        Matrix containing the threholded fiducials
    fiducial_points_found: list (2d)
        The locations of the fiducials
    fiducial_cm: tuple
        The location of the center of mass of all the fiducials
    '''
    fiducial_matrix = np.zeros(img_shape, dtype=np.uint16)
    fiducial_points_found = []
    for point in fiducial_points_guess : 
        
        # create non-zero shpere around fiducial guess in which to threshold
        center = cm_to_index(point, origin, pixel_spacing)
        print(center)
        distance = np.linalg.norm(np.subtract(np.indices(img_shape).T,np.asarray(center)), axis=len(center)).T
        distance = distance.astype(np.uint16)
        mask = np.ones(img_shape, dtype=np.uint16) * (distance<=fiducial_rad) * img3d

        # threshold to find fiducial
        thresh = exam.Series[0].ImageStack.MaxStoredValue / 2
        mask[mask<=thresh] = 0

        # add fiducial center of mass
        fiducial_points_found.append(center_of_mass(mask))

        # add fiducial to fiducial_matrix
        fiducial_matrix += mask

    #fiducial center of mass
    fiducial_cm = center_of_mass(fiducial_matrix)

    return fiducial_matrix, fiducial_points_found, fiducial_cm

def get_search_radius (fiducial_points_guess, origin, pixel_spacing):
    '''
    get_search_radius takes the guess lotation for the fiducial points, and dertermines
    and appropriate radius for searching. The search radius is the half the 
    minimum distance between fiducials guess. This ensures that two fiducials are not confused.


    Parameters
    ----------
    fiducial_points_guess : list (2d) [[z,y,x], ... ]
        Guesses for the fiducial locations in cm
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm

    Returns
    -------
    fiducial_rad : float
        Radius of search for fiducials
    '''
    if len(fiducial_points_guess) < 2 :
        return 10
    fid_point_index = []
    distances = []
    for val, point in enumerate(fiducial_points_guess) : 
        fid_point_index.append(np.array(cm_to_index(point, origin, pixel_spacing)))
        if val > 0 : 
            squared_dist = np.sum((fid_point_index[val]-fid_point_index[val-1])**2, axis=0)
            distances.append(np.sqrt(squared_dist))

    fiducial_rad = float(min(distances)) / 2 
    return fiducial_rad

###################################################################
#################### poi functions ################################
###################################################################

def get_poi_data (case, exam) : 
    '''
    get_poi_data takes a case and exam, and returns lists of the names and 
    locations of the POIs in that exam.

    Parameters
    ----------
    case : PyScriptObject
        case data from the raystation statetree
    exam : PyScriptObject
        examination data from the raystation statetree

    Returns
    -------
    poi_names : list
        Names on POIs in selected case and exam
    poi_locations : list of tuples [(z_1,y_1,x_1), ...]
        z, y, and x locations of all POIs in selected case and exam
    '''
    # create lists
    poi_locations = []
    poi_names = []
    # determine locations of all POIs in case/examination
    for poi in case.PatientModel.StructureSets[exam.Name].PoiGeometries : 
        poi_locations.append((poi.Point.z, poi.Point.y, poi.Point.x))
        poi_names.append(poi.OfPoi.Name)
    return poi_locations, poi_names

def move_pois (case, exam, fiducial_points, poi_names, origin, pixel_spacing) : 
    '''
    move_pois moves existing pois to new found fiducial locations.

    Parameters
    ----------
    case : PyScriptObject
        case data from the raystation statetree
    exam : PyScriptObject
        examination data from the raystation statetree
    fiducial_points : list (2d)
        The locations of the fiducials
    poi_names : list
        names on POIs in selected case and exam
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm

    Returns
    -------
    None
    '''
    # check that the number of poi names and location match
    if len(fiducial_points) != len(poi_names) : 
        print("The number of fiducial points and names don't match")
        return None
    
    # change the positions of the POIs
    for i, name  in enumerate(poi_names) : 
        point_cm = index_to_cm (fiducial_points[i], origin, pixel_spacing)
        new_point = {'x' : point_cm[2], 'y' : point_cm[1], 'z' : point_cm[0]}
        case.PatientModel.StructureSets[exam.Name].PoiGeometries[name].Point = new_point

    return None 

def create_pois (case, exam, fiducial_points, poi_names, origin, pixel_spacing) : 
    '''
    create_pois creates new POIS from found fiducial locations and a list of names.

    Parameters
    ----------
    case : PyScriptObject
        case data from the raystation statetree
    exam : PyScriptObject
        examination data from the raystation statetree
    fiducial_points : list (2d)
        The locations of the fiducials
    poi_names : list
        Names on POIs in selected case and exam
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm

    Returns
    -------
    None
    '''
    # check that the number of poi names and location match
    if len(fiducial_points) != len(poi_names) : 
        print("The number of fiducial points and names don't match")
        return None

    for i, name  in enumerate(poi_names) : 
        point_cm = index_to_cm (fiducial_points[i], origin, pixel_spacing)
        new_point = {'x' : point_cm[2], 'y' : point_cm[1], 'z' : point_cm[0]}
        case.PatientModel.CreatePoi(Examination=exam, Point=new_point, Volume=0, Name=name, Color="Yellow", Type="Undefined")

    return None

def new_poi_geometries (case, exam, fiducial_points, poi_names, origin, pixel_spacing) : 
    '''
    new_poi_geometries creates new POI geometries on a given case and exam 
    for existing POIs.

    Parameters
    ----------
    case : PyScriptObject
        case data from the raystation statetree
    exam : PyScriptObject
        examination data from the raystation statetree
    fiducial_points : list (2d)
        The locations of the fiducials
    poi_names : list
        Names on POIs in selected case and exam
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm

    Returns
    -------
    None
    '''
    # check that the number of poi names and location match
    if len(fiducial_points) != len(poi_names) : 
        print("The number of fiducial points and names don't match")
        return None

    for i, name  in enumerate(poi_names) :
        point_cm = index_to_cm (fiducial_points[i], origin, pixel_spacing)
        new_point = {'x' : point_cm[2], 'y' : point_cm[1], 'z' : point_cm[0]}
        case.PatientModel.StructureSets[exam.Name].PoiGeometries[name].Point = new_point

    return None

###################################################################
#################### math functions ###############################
###################################################################

def index_to_cm (point_index, origin, pixel_spacing, rounding=False):
    '''
    index_to_cm converts a point position in index values of the numpy array to positions in cm. 
    All values are rounded to nearest integer. 

    Parameters
    ----------
    point_index : list [z_index,y_index,x_index]
        Point index values to convert
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm

    Returns
    -------
    point_cm : list [z_cm,y_cm,x_cm]
        Point values in cm
    '''
    x_cm = (origin[2] + point_index[2] * pixel_spacing[2])
    y_cm = (origin[1] + point_index[1] * pixel_spacing[1])
    z_cm = (origin[0] + point_index[0] * pixel_spacing[0])

    if rounding : 
        return [round(z_cm), round(y_cm), round(x_m)]

    return [z_cm, y_cm, x_cm]

def cm_to_index (point_cm, origin, pixel_spacing, rounding=True):
    '''
    index_to_cm converts a point position in cm to positions in index values.
    All values are rounded to nearest integer. 

    Parameters
    ----------
    point_cm : list [z_cm,y_cm,x_cm]
        Point values in cm tp convert
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in cm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in cm

    Returns
    -------
    point_index : list [z_index,y_index,x_index]
        Point index values 
    '''
    x_index = ((point_cm[2] - origin[2]) / pixel_spacing[2])
    y_index = ((point_cm[1] - origin[1]) / pixel_spacing[1])
    z_index = ((point_cm[0] - origin[0]) / pixel_spacing[0])

    if rounding : 
        return [round(z_index), round(y_index), round(x_index)]

    return [z_index, y_index, x_index]

def center_of_mass(arr):
    """
    Calculate the center of mass of the values of an array at labels.
    Parameters
    ----------
    arr : ndarray
        Data from which to calculate center-of-mass.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.
    """
    normalizer = np.sum(arr)
    grids = np.ogrid[[slice(0, i) for i in arr.shape]]

    results = [np.sum(arr * grids[num].astype(float)) / normalizer for num in range(arr.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]


###################################################################
#################### define program ###############################
###################################################################


def main () : 
    # get current examination loaded as primary for current case and patient
    # requires that patient is open and that there exisits a case and examination
    try : 
        patient = get_current("Patient")
        case = get_current("Case")
        examination = get_current("Examination")
    except : 
        print ("patient, case, and examination not open")
        sys.exit()

    # get poi locations in reference image (TPCT)
    poi_locations, poi_names = get_poi_data (case, examination)
    fiducial_points_guess = poi_locations

    for i, exam in enumerate(case.Examinations) :
        # get the ct as numpy array, and associated shape values
        img3d, img_shape, origin, pixel_spacing = image_data(exam)
        # determine seach radius from starting search locations
        fiducial_rad = get_search_radius(fiducial_points_guess, origin, pixel_spacing)
        # get fiducials and print 
        fiducial_matrix, fiducial_points_found, fiducial_cm = fiducial_search_area(exam, img3d, fiducial_points_guess, fiducial_rad, img_shape, origin, pixel_spacing)
        # create new POIs where fiducials were found 
        if exam.Name == examination.Name : 
            move_pois(case, exam, fiducial_points_found, poi_names, origin, pixel_spacing)
        else : 
            new_poi_geometries(case, exam, fiducial_points_found, poi_names, origin, pixel_spacing)

    return

if __name__ == "__main__":
    main()
