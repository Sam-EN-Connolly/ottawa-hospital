import sys
import numpy as np
from connect import *

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
        Names on POIs in selected case and exam

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
