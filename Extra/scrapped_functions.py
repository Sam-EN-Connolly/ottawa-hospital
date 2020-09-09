'''
Functions used in development that have been abandoned 
'''
import numpy as np
import pydicom as dcm
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
import os
import glob

def get_ct_matrix (exam, img_shape) :
    '''
    get_ct_matrix loads the current examination ct image into a numpy array.

    parameters : 
    exam (ScriptObject) : contains the current examination data from the raystation statetree
    img_shape (list) : the shape of the ct image in pixels/indecies of form [z,y,x]

    returned values :
    img3d (3d numpy array) : matrix containing the pixel values of the ct image
    '''
    # get pixel data from raystation state tree
    byte_array = exam.Series[0].ImageStack.PixelData
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


def get_paths(folder_path):
    logfile_path = False
    plan_path = False
    for root, dirs, files in os.walk(folder_path, topdown=False): 
        if not logfile_path or not plan_path : 
            for f in files : 
                if f == 'AlgorithmImaging.log' : 
                    print("here1")
                    logfile_path = os.path.join(root, f)
                if '_plan.xml' in f :  
                    print("here2")
                    plan_path = os.path.join(root, f)
    return logfile_path, plan_path

def reference_fid_values (logfile_path) : 
    '''
    reference_fid_values reads one of the extracted AlgorithImaging logfiles
    and extracts the positions of the reference fiducials in the TPCT.

    works for log files. 

    Parameters
    ----------
    logfile_path : string
        Path to directory containing logfile

    Returns
    -------
    fiducial_locations : list (2d)
        list of locations of fiducials in TPCT from logfile. 
        the fiducial locations are given in lists of [z,y,x].
    '''
    fiducial_lines = False
    fiducial_locations = []
    with open(logfile_path, 'r') as logfile : 
        contents = logfile.readlines()
        for line in contents : 
            if not fiducial_lines : 
                if 'Fid' in line and 'Reference' in line : 
                    fiducial_lines = True
            elif fiducial_lines :
                if line == '\n' : 
                    break
                location = reversed(line.split()[-1].strip('()').split(','))
                fiducial_locations.append([float(i) for i in location])
    
    return fiducial_locations


###################################################################
#################### display functions ############################
###################################################################

def plot_fiducials(img3d, fiducial_matrix, fiducial_points_found, origin, pixel_spacing) : 
    '''
    plot_fiducials plots the slices containing fiducials. 

    Parameters
    ----------
    img3d : ndarray (3d)
        Matrix containing the pixel values of the ct image
    fiducial_matrix : ndarray (3d) 
        Matrix containing the threholded fiducials
    fiducial_points_found: list (2d)
        The locations of the fiducials
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    None 

    The inputs are the 3D matrix containing the image data (img3d), 
    the 3d numpy matrix containing the threholded fiducials (fiducial_matrix),
    and the locations of the fiducials to plot (fiducial_points_found).
    '''

    fiducial_points_mm = []
    for point in fiducial_points_found : 
        fiducial_points_mm.append(index_to_mm(point, origin, pixel_spacing))

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
    return None


def get_contour_points (ct_slice, rts, contour_points, roi_contour_sequence_index, SOP_UIDs, img_shape, origin, pixel_spacing) : 
    '''
    get_contour_points finds the contour points in a slice and appeds them to contour_points.
    python passes lists by reference. 

    Parameters
    ----------
    ct_slice : <class 'pydicom.dataset.FileDataset'>
        one ct slice from scan
    rts : file
        The open rtss file with contours
    contour_points : list
        list of lists on contours points in given slice. if no contour points in that slice 
        points are given as [np.nan, np.nan]. this list is passed by reference and modified within th function.
    roi_contour_sequence_index : int
        Index for ROIContourSequence that holds contour of interest
    SOP_UIDs : list
        Lsit of SOP UIDs with contour of interest
    img_shape : tuple (z,y,x)
        Shape of img3d in mm 
    origin : typle (z,y,x)
        Location of image origin relative to numpy matrix indexing in mm
    pixel_spacing : tuple (z,y,x)
        Distance between the centre of pixel in all three dimentions in mm

    Returns
    -------
    None 

    '''
    # get y_origin
    y_origin = origin[1]

    #fill associated structure matrix
    if ct_slice.SOPInstanceUID in SOP_UIDs:
            
        contour_points_x = [[float(y) for y in x.ContourData[0::3]] for x in rts.ROIContourSequence[roi_contour_sequence_index].ContourSequence\
                            if x.ContourImageSequence[0].ReferencedSOPInstanceUID == ct_slice.SOPInstanceUID ][0]
        contour_points_y = [[float(y) for y in x.ContourData[1::3]] for x in rts.ROIContourSequence[roi_contour_sequence_index].ContourSequence\
                            if x.ContourImageSequence[0].ReferencedSOPInstanceUID == ct_slice.SOPInstanceUID ][0]
        
        #convert contour points y for orientaton (currently assumed HF supine!!!!)
        contour_points_y = [y_origin + img_shape[1] * pixel_spacing[1] - (y-y_origin)  for y in contour_points_y]
                
        #append first point of contour to close structure if not already done (manual structures have this already)
        if contour_points_x[0] != contour_points_x[-1] or contour_points_y[0] != contour_points_y[-1]:
            contour_points_x.append(contour_points_x[0])
            contour_points_y.append(contour_points_y[0])

        contour_points.append([contour_points_x, contour_points_y])

    else:
        contour_points.append([np.nan, np.nan])   

def get_contour_dict(rts_path, structure_wanted): 
    '''Given the path of a dicom structure file, return the contour information
    as a dictionary where indexes are the names of the contours'''
    rts = dcm.dcmread(rts_path, force = True)
    ##Find the structure set objects that are to be compared
    num2name = dict()
    for structureSet in rts.StructureSetROISequence:
        num2name[structureSet.ROINumber] = structureSet.ROIName
    
    contours = dict()
    for contour in rts.ROIContourSequence:
        num = contour.ReferencedROINumber
        contours[num2name[num]] = contour.ContourSequence

    contour_dict = defaultdict(list)

    c = contours[structure_wanted]
    for contourSlice in c:
        contour_data = np.array(contourSlice.ContourData)

        contour_data = contour_data.reshape((len(contour_data)//3,3))
        z = round(contour_data[2,2])
        contour_dict[z].append(contour_data[:,0:2])
        
    return contour_dict


def calc_similarity_metrics(contour_1, contour_2, structures_wanted):

    # check if structure in structure sets
    structure_1_found = False
    structure_2_found = False
    structure_wanted_list = []

    for structure in structures_wanted : 
        if structure in contour_1.keys() and not structure_1_found : 
            structure_wanted_list.append(structure)
            structure_1_found = True
        if structure in contour_2.keys() and not structure_2_found : 
            structure_wanted_list.append(structure)
            structure_2_found = True
    if not (structure_1_found or structure_2_found) : 
        print("Missing structure set")
        return None, None, None, None

    margin = 10
    contours = [contour_1, contour_2]
    
    indexmin = np.ones(3)*np.inf
    indexmax = -np.ones(3)*np.inf
    
    contourDicts =[defaultdict(list),defaultdict(list)]
    zunique = set()
    # get contour points and reshape inton ndarray for plotting
    for i, contourSequence in enumerate(contours):
        c = contourSequence[structure_wanted_list[i]]
        for contourSlice in c:
            contourData = np.array(contourSlice.ContourData)
            contourData = contourData.reshape((len(contourData)//3,3))
            z = round(contourData[2,2])
            zunique.add(z)
            contourDicts[i][z].append(contourData[:,0:2])
            # get minimum and maximum indecies to plot within 
            indexmin = np.minimum(contourData.min(axis = 0),indexmin)
            indexmax = np.maximum(contourData.max(axis = 0),indexmax)
    
    
    x =(np.ceil(indexmax)-np.floor(indexmin)+2*margin) # this should be in mm
    x = [int(i) for i in x]
    zeroIdx = indexmin[0:2].astype(np.int32)-margin
    
    set_1 = 0
    set_2 = 0 
    set_overlap = 0
    slices1 = list()
    slices2 = list()

    for z in sorted(list(zunique)):

        # determine areas and arrays of both contours and append to respective 3d matrix
        img1 = plot_contours(x,zeroIdx,contourDicts[0][z])
        img2 = plot_contours(x,zeroIdx,contourDicts[1][z])
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
    
    return msd, rms, hd, dsc


def get_contour_data(ct_folders, workbook_out, csv_out) : 
    # Create a workbook for saving data
    workbook = xlsxwriter.Workbook(workbook_out)

    # to hold contour names and image set names
    contour_names = []

    # create list for combined out data
    list_dict_out = []
    
    # define initial variables
    file_names = [fname for fname in os.listdir(ct_folders) if not fname.startswith('.')]
    structures_wanted = ['Prostate', 'prostate', 'PROSTATE', 'PROS1_3625']

    # go through all images in data base
    for MRN in file_names:
        print(MRN)
        rts_folder = os.path.join(ct_folders, MRN, 'RTS/')
        rts_files = glob.glob(os.path.join(rts_folder,'*.dcm'), recursive=None)
        rts_dicts = []

        for file_path in rts_files:
            contours, contour_name, image_set = convert_to_dict(file_path)
            contour_dict, zunique, indexmin, indexmax, ROI_COM = get_contour_points_dict(contours, structures_wanted)
            rts_dicts.append((contour_dict, zunique, indexmin, indexmax))
            contour_names.append(str(contour_name) + ", " + str(image_set))

        msd1, rms1, hd1, dsc1 = calc_similarity_metrics(rts_dicts[0][0], rts_dicts[0][1], rts_dicts[0][2], rts_dicts[0][3], rts_dicts[1][0], rts_dicts[1][1], rts_dicts[1][2], rts_dicts[1][3])
        msd2, rms2, hd2, dsc2 = calc_similarity_metrics(rts_dicts[0][0], rts_dicts[0][1], rts_dicts[0][2], rts_dicts[0][3], rts_dicts[2][0], rts_dicts[2][1], rts_dicts[2][2], rts_dicts[2][3])
        msd3, rms3, hd3, dsc3 = calc_similarity_metrics(rts_dicts[1][0], rts_dicts[1][1], rts_dicts[1][2], rts_dicts[1][3], rts_dicts[2][0], rts_dicts[2][1], rts_dicts[2][2], rts_dicts[2][3])
        
        # add data to list for saving to csv for future analysis
        patient_dict = {'MRN' : MRN,
                        'Primanry Image' : contour_names[0],
                        'Comparison 1' : contour_names[1],
                        'Mean Surface Distance 1' : round(msd1,2),
                        'Residual Mean-Square Error 1' : round(rms1,2),
                        'Hausdorff Distance 1' : round(hd1,2),
                        'Dice coefficient 1' : round(dsc1,2),
                        'Comparison 2' : contour_names[2],
                        'Mean Surface Distance 2' : round(msd2,2),
                        'Residual Mean-Square Error 2' : round(rms2,2),
                        'Hausdorff Distance 2' : round(hd2,2),
                        'Dice coefficient 2' : round(dsc2,2)}
        list_dict_out.append(patient_dict)


        # create new worksheet for surrent patient
        worksheet = workbook.add_worksheet(MRN)

        # write data to worksheet
        worksheet.write_string(0, 0, "Comparision 1")
        worksheet.write_string(1, 0, "Contour 1") 
        worksheet.write(1, 1, contour_names[0])
        worksheet.write_string(2, 0, "Contour 2 : ") 
        worksheet.write(2, 1, contour_names[1])
        worksheet.write_string(3, 0, "Mean Surface Distance") 
        worksheet.write(3, 1, round(msd1,2))
        worksheet.write_string(4, 0, "Residual Mean-Square Error")
        worksheet.write(4, 1, round(rms1,2))
        worksheet.write_string(5, 0, "Hausdorff Distance") 
        worksheet.write(5, 1, round(hd1,2))
        worksheet.write_string(6, 0, "Dice Coefficient")
        worksheet.write(6, 1, round(dsc1,2))


        worksheet.write(8, 0, "Comparision 2")
        worksheet.write_string(9, 0, "Contour 1") 
        worksheet.write(9, 1, contour_names[0])
        worksheet.write_string(10, 0, "Contour 2 : ") 
        worksheet.write(10, 1, contour_names[2])
        worksheet.write_string(11, 0, "Mean Surface Distance") 
        worksheet.write(11, 1, round(msd2,2))
        worksheet.write_string(12, 0, "Residual Mean-Square Error")
        worksheet.write(12, 1, round(rms2,2))
        worksheet.write_string(13, 0, "Hausdorff Distance") 
        worksheet.write(13, 1, round(hd2,2))
        worksheet.write_string(14, 0, "Dice Coefficient")
        worksheet.write(14, 1, round(dsc2,2))

        worksheet.write(16, 0, "Comparision 3")
        worksheet.write_string(17, 0, "Contour 1") 
        worksheet.write(17, 1, contour_names[1])
        worksheet.write_string(18, 0, "Contour 2 : ") 
        worksheet.write(18, 1, contour_names[2])
        worksheet.write_string(19, 0, "Mean Surface Distance") 
        worksheet.write(19, 1, round(msd3,2))
        worksheet.write_string(20, 0, "Residual Mean-Square Error")
        worksheet.write(20, 1, round(rms3,2))
        worksheet.write_string(21, 0, "Hausdorff Distance") 
        worksheet.write(21, 1, round(hd3,2))
        worksheet.write_string(22, 0, "Dice Coefficient")
        worksheet.write(22, 1, round(dsc3,2))
    
    workbook.close()
    # write data to csv file
    write_csv_from_dict(csv_out, list_dict_out)


def fiducial_change_data(csv_file_path_found_locations, csv_file_path_fiducial_change, ROI_dict) : 
    list_dict_in = read_csv_to_list_dict(csv_file_path_found_locations)
    dict_out = {}
    list_dict_out = []

    for patient in list_dict_in : 
        # get data for current patient
        center_of_mass_TPCT = patient['Fiducial COM TPCT']
        center_of_mass_Fused = patient['Fiducial COM Fused']
        fiducials_TPCT = patient['Fiducial Positions TPCT']
        fiducials_Fused = patient['Fiducial Positions Fused']

        # determine the average distance to the fiducials and the change in the average distance
        avg_dist_fidCOM_TPCT = average_fid_distance(fiducials_TPCT, center_of_mass_TPCT)
        avg_dist_fidCOM_Fused = average_fid_distance(fiducials_Fused, center_of_mass_Fused)
        change_dist_fidCOM = avg_dist_fidCOM_TPCT - avg_dist_fidCOM_Fused

        avg_dist_ROI_TPCT = average_fid_distance(fiducials_TPCT, ROI_dict[patient['MRN']][0])
        avg_dist_ROI_Fused = average_fid_distance(fiducials_Fused, ROI_dict[patient['MRN']][2])
        change_dist_ROI = avg_dist_ROI_TPCT - avg_dist_ROI_Fused
        scaling_factor = get_iso_scale_factor(avg_dist_ROI_TPCT, avg_dist_ROI_Fused)

        # determine change in each fiducial position
        position_change = change_fid_positions(fiducials_TPCT, fiducials_Fused)

        patient_dict = {'Change in fiducial locations [fid1, fid2,...]' : position_change,
                        'Change in average distance to fid_COM' : change_dist_fidCOM,
                        'Change in average distance to ROI_COM' : change_dist_ROI,
                        'Fiducial COM TPCT' : center_of_mass_TPCT,
                        'Fiducial COM Fused' : center_of_mass_Fused,
                        'ROI center TPCT' : ROI_dict[patient['MRN']][0],
                        'ROI center Fused' : ROI_dict[patient['MRN']][2],
                        'Volume TPCT' : ROI_dict[patient['MRN']][1],
                        'Volume Fused' : ROI_dict[patient['MRN']][3],
                        'Isotropic scaling factor' : scaling_factor}
        dict_out[patient['MRN']] = patient_dict
        patient_dict['MRN'] = patient['MRN']
        list_dict_out.append(patient_dict)
    
    write_csv_from_dict(csv_file_path_fiducial_change, list_dict_out)

    return dict_out

def read_csv_to_list_dict(csv_file_path) : 
    '''
    read_fid_location_csv reads the csv file containing the reference fiducial locations,
    with two columns, and returns a dictionsary where the keys are column 1 and 
    the associated values are column two. 
    This functions reads the out write_csv_from_dict for the reference 
    fiducial locations.

    Parameters
    ----------
    csv_file_path : string
        path and name of csv file to read

    Returns
    -------
    fiducial_dict : dictionary
        fiducial location data from csv file. keys are MRN and values
        are fiducial locations as a list. 
    '''
    csv_list = []
    with open(csv_file_path, 'r') as csv_file : 
        csv_reader = csv.DictReader(csv_file)
        keys = csv_reader.fieldnames
        for line in csv_reader : 
            working_dict = {}
            for key in keys : 
                try : 
                    working_dict[key] = eval(line[key])
                except SyntaxError: 
                    working_dict[key] = line[key]
            csv_list.append(working_dict)

    return csv_list


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

    # repeat for signed values 
    S_signed = np.subtract((~input_1).astype(np.int), (morphology.binary_erosion(input_1, conn)).astype(np.int))
    Sprime_signed = np.subtract((~input_2).astype(np.int), (morphology.binary_erosion(input_2, conn)).astype(np.int))

    # create signed distance feild
    dta_signed = skfmm.distance(S_signed)
    dtb_signed = skfmm.distance(Sprime_signed)

    # create and flatten surface distance array 
    sds_signed = np.concatenate([np.ravel(dta_signed[Sprime_signed==0]), np.ravel(dtb_signed[S_signed==0])])
    

    return sds