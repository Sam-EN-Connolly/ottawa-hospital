'''
Sam Connolly
Version 1: June 4, 2020
Contact: saconnolly@toh.ca 

This program reads the current open examination image data and finds the location of the fiducials.
Currently the fidicuial search locations are hard-coded in for testing purposes. 
It requires that a patien, case, and examination are open. 
'''

from connect import *
import sys
import numpy as np

###################################################################
#################### image extraction #############################
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
        Shape of img3d in mm 
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm
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

    # determine the resclae values to turn pixel values to HU values
    #rescale_intercept = exam.Series[0].ImageStack.ConversionParameters.RescaleIntercept
    #rescale_slope = exam.Series[0].ImageStack.ConversionParameters.RescaleSlope
    
    # determine if data type of pixel, unsigned integer or two's compliment
    pixel_representation = exam.Series[0].ImageStack.ConversionParameters.PixelRepresentation

    # get pixel data from RayStation
    pixel_data = exam.Series[0].ImageStack.PixelData
    # convert from 16-bit to integers
    length = len(pixel_data)
    evens = np.arange(0, length, 2, dtype=np.uint16)
    odds = np.arange(1, length, 2, dtype=np.uint16)
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

###################################################################
#################### fiducial search ##############################
###################################################################

def fiducial_search_area (img3d, fiducial_points_guess, fiducial_rad, img_shape, origin, pixel_spacing) :
    '''
    fiducial_search_area takes guesses for the fiducial locations and searhes for them within a certain radius through thresholding. 

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
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    fiducial_matrix : ndarray (3d) 
        Matrix containing the threholded fiducials
    fiducial_points_found: list (2d)
        The locations of the fiducials
    fiducial_cm: tuple
        The location of the center of mass of all the fiducials
    '''
    fiducial_matrix = np.zeros(img_shape)
    fiducial_points_found = []
    for point in fiducial_points_guess : 
        
        # create non-zero shpere around fiducial guess in which to threshold
        center = mm_to_index(point, origin, pixel_spacing)
        distance = np.linalg.norm(np.subtract(np.indices(img_shape).T,np.asarray(center)), axis=len(center)).T
        mask = np.ones(img_shape) * (distance<=fiducial_rad) * img3d

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
    fiducial_points_guess : list [z,y,x]
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
    fid_point_index = []
    distances = []
    for val, point in enumerate(fiducial_points_guess) : 
        fid_point_index.append(np.array(mm_to_index(point, origin, pixel_spacing)))
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
    for name in case.PatientModel.StructureSets[exam.Name].PoiGeometries : 
        point = dict(case.PatientModel.StructureSets[exam.Name].PoiGeometries[name].Point)
        poi_locations.append((point['z'], point['y'], point['x']))
        poi_names.append(name)
    return poi_locations, poi_names

def move_pois (case, exam, fiducial_points, poi_names) : 
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

    Returns
    -------
    None
    '''
    # check that the number of poi names and location match
    if len(fiducial_points) != len(poi_names) : 
        print("The number of fiducial points and names don't match")
        return None
    
    # change the positions of the 
    for i, name  in enumerate(poi_names) : 
        new_point = {'x' : fiducial_points[i][2], 'y' : fiducial_points[i][1], 'z' : fiducial_points[i][0]}
        case.PatientModel.StructureSets[exam.Name].PoiGeometries[name].Point = new_point

    return None 

def create_pois (case, exam, fiducial_points, poi_names) : 
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

    Returns
    -------
    None
    '''
    # check that the number of poi names and location match
    if len(fiducial_points) != len(poi_names) : 
        print("The number of fiducial points and names don't match")
        return None

    for i, name  in enumerate(poi_names) : 
        new_point = {'x' : fiducial_points[i][2], 'y' : fiducial_points[i][1], 'z' : fiducial_points[i][0]}
        case.PatientModel.CreatePoi(Examination=exam, Point=new_point, Volume=0, Name=name, Colour="Yellow", Type="Undefined")

    return None

###################################################################
#################### math functions ###############################
###################################################################

def index_to_mm (point_index, origin, pixel_spacing, rounding=False):
    '''
    index_to_mm converts a point position in index values of the numpy array to positions in mm. 
    All values are rounded to nearest integer. 

    Parameters
    ----------
    point_index : list [z_index,y_index,x_index]
        Point index values to convert
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    point_mm : list [z_mm,y_mm,x_mm]
        Point values in mm
    '''
    x_mm = (origin[0] + point_index[2] * pixel_spacing[2])
    y_mm = (origin[1] + point_index[1] * pixel_spacing[1])
    z_mm = (origin[0] + point_index[0] * pixel_spacing[0])

    if rounding : 
        return [round(z_mm), round(y_mm), round(x_m)]

    return [z_mm, y_mm, x_mm]

def mm_to_index (point_mm, origin, pixel_spacing, rounding=True):
    '''
    index_to_mm converts a point position in mm to positions in index values.
    All values are rounded to nearest integer. 

    Parameters
    ----------
    point_mm : list [z_mm,y_mm,x_mm]
        Point values in mm tp convert
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    point_index : list [z_index,y_index,x_index]
        Point index values 
    '''
    x_index = ((point_mm[2] - origin[2]) / pixel_spacing[2])
    y_index = ((point_mm[1] - origin[1]) / pixel_spacing[1])
    z_index = ((point_mm[0] - origin[0]) / pixel_spacing[0])

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


###################################################################
#################### testing functions ############################
###################################################################


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
    poi_locations, poi_names = get_poi_data (case, exam)
    fiducial_points_guess = poi_locations

    for exam in case.Examinations :
        # get the ct as numpy array, and associated shape values
        img3d, img_shape, origin, pixel_spacing = image_data(exam)
        # determine seach radius from starting search locations
        fiducial_rad = get_search_radius(fiducial_points_guess, origin, pixel_spacing)
        # get fiducials and print 
        fiducial_matrix, fiducial_points_found, fiducial_cm = fiducial_search_area(img3d, fiducial_points_guess, fiducial_rad, img_shape, origin, pixel_spacing)
        # create new POIs where fiducials were found 
        if exam.Name == examination.Name : 
            move_pois (case, exam, fiducial_points_found, poi_names)
        else : 
            create_pois (case, exam, fiducial_points, poi_names)

    return

if __name__ == "__main__":
    main()
