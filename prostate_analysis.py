import os

from functions_fiducials import *
from functions_fiducial_location_properties import *
from functions_fileIO import *

'''
prostate_analysis.py contains a collections of functions to run different analyses on 
the prostate batabase. The functions are mainly written to be able to operate separately
so that different aspects of the analysis can be re-run independantly. This causes signifigant 
code replication. To improve the efficiancy of this code, functions would need to be combined 
such that data is not saved and re-read multipled times throughout the entire analysis. 

Each functions contains a description of the analysis actions performed. 
All imported functions are contained within the the same folder as this file. 
'''

def extract_data(csv_reference_locations, csv_metadata, csv_fiducial, ct_folders, metatdata_dict, fiducial_dict) :
    '''
    extract_data extracts the images and relevent metada from the DICOM files. 
    It also finds the fiducial location in both the TPCT and Fused images. 
    The fiducial search locations are determined by the reference locations in the 
    treatment logfiles, which must have already been extracted and saved to fiducial_reference_locations.csv.
    The fiducial location is found by searching in a radius around the reference location, 
    and performing a thresholding on this area of the image. The center of mass (or brightness)
    is then set as the fiducial location. This extracted data is then saved in a csv files 
    named metadata.csv and fiducial.csv. The data is also returned in dictionaries for further analysis.

    Parameters
    ----------
    csv_reference_locations : String
        path to saved fiducial locations from previouslyextracted logfiles
    csv_metadata : String
        path for where to save metadata information, to be read for subsequent running of program
    csv_fiducial : String
        path for where to save fidcual location information, to be read for subsequent running of program
    ct_folders : String
        path to folder containing patients CT image and rts files
    metatdata_dict : nested dictionary 
        Passed in empty (metadata_dict = {})
    fiducial_dict : nested dictionary 
        Passed in empty (metadata_dict = {}) 
    Returns
    -------
    metatdata_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED'
        Third level of dictionary hold the variables with keys : 'image shape', 'origin', 'pixel spacing'
    fiducial_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED' and 'CHANGE'
        Third level of dictionary hold the variables with keys : 'locations', 'COM'

    '''
    # read in reference locations for fiducials
    fiducial_reference_dict = read_reference_location_csv(csv_reference_locations)
    # get all file names (MRN) 
    file_names = [fname for fname in os.listdir(ct_folders) if not fname.startswith('.')]
    # iniitialze list for writing to csv
    out_list_fiducials = []
    out_list_metadata = []

    # go through all images in data base
    for MRN in file_names:
        # get guess location for fiducial
        fiducial_points_guess = fiducial_reference_dict[MRN]
        # initialize working dicts
        metatdata_dict[MRN] = {'TPCT' : None, 'FUSED' : None}
        fiducial_dict[MRN] = {'TPCT' : None, 'FUSED' : None, 'CHANGE' : None}

        # data analysis for TPCT
        dcm_files_TPCT = open_dcm_files(os.path.join(ct_folders, MRN, 'TPCT/'))
        img3d_TPCT, img_shape_TPCT, origin_TPCT, pixel_spacing_TPCT = dcm_to_3d_arr(dcm_files_TPCT)
        fiducial_rad_TPCT = get_search_radius(fiducial_points_guess, origin_TPCT, pixel_spacing_TPCT)
        # find fiducials
        fiducial_matrix_TPCT, fiducial_points_found_TPCT, fiducialCM_TPCT = fiducial_search_area(img3d_TPCT, fiducial_points_guess, img_shape_TPCT, origin_TPCT, fiducial_rad_TPCT, pixel_spacing_TPCT)
        
        # data analysis for fused image
        dcm_files_Fused = open_dcm_files(os.path.join(ct_folders, MRN, 'Fused/'))
        img3d_Fused, img_shape_Fused, origin_Fused, pixel_spacing_Fused = dcm_to_3d_arr(dcm_files_Fused)
        fiducial_rad_Fused = get_search_radius(fiducial_points_guess, origin_Fused, pixel_spacing_Fused)
        # find fiducials
        fiducial_matrix_Fused, fiducial_points_found_Fused, fiducialCM_Fused = fiducial_search_area(img3d_Fused, fiducial_points_guess, img_shape_Fused, origin_Fused, fiducial_rad_Fused, pixel_spacing_Fused)

        out_dict_metadata = {'MRN' : MRN,
                    'Image Shape TPCT' : img_shape_TPCT,
                    'Origin TPCT' : origin_TPCT, 
                    'Pixel Spacing TPCT' : pixel_spacing_TPCT,
                    'Image Shape Fused' : img_shape_Fused,
                    'Origin Fused' : origin_Fused, 
                    'Pixel Spacing Fused' : pixel_spacing_Fused}

        
        out_dict_fiducials = {'MRN' : MRN,
                    'Fiducial Positions Reference' : fiducial_points_guess,
                    'Fiducial Positions TPCT' : fiducial_points_found_TPCT.tolist(), 
                    'Fiducial COM TPCT' : fiducialCM_TPCT.tolist(),
                    'Fiducial Positions Fused' : fiducial_points_found_Fused.tolist(),
                    'Fiducial COM Fused' : fiducialCM_Fused.tolist()}

        out_list_metadata.append(out_dict_metadata)
        out_list_fiducials.append(out_dict_fiducials)

        # add data to working dictionaries (metatdata and fiducials)
        # TPCT
        metatdata_dict[MRN]['TPCT'] = {'image shape' : img_shape_TPCT, 'origin' : origin_TPCT, 'pixel spacing' : pixel_spacing_TPCT}
        fiducial_dict[MRN]['TPCT'] = {'locations' : fiducial_points_found_TPCT, 'COM' : fiducialCM_TPCT}
        # Fused
        metatdata_dict[MRN]['FUSED'] = {'image shape' : img_shape_Fused, 'origin' : origin_Fused, 'pixel spacing' : pixel_spacing_Fused}
        fiducial_dict[MRN]['FUSED'] = {'locations' : fiducial_points_found_Fused, 'COM' : fiducialCM_Fused}

    # write data to csv file
    write_csv_from_dict(csv_metadata, out_list_metadata)
    write_csv_from_dict(csv_fiducial, out_list_fiducials)

    return metatdata_dict, fiducial_dict

def review_contour_data(ct_folders, workbook_out, structures_wanted) : 
    '''
    review_contour_data goes through all the contour structure files and saves the comparison metrics to an excel file.
    It compared the human-done contour to the two auto-contours (one for the TPCT and one for the Fused image).
    The mean surface distance, residual mean-square error, hausdorff distance, and dice coefficiant for 
    each comparison are saved to a file called contour_data.xlsx.

    Parameters
    ----------
    ct_folders : String
        path to folder containing patients CT image and rts files
    workbook_out : 
        path to excel file to be saved 
    structures_wanted : list
        List of strings, with possible names for region of interest to be examined
        Eg ["PROSTATE", "prostate", "Prostate"]

    Returns
    -------
    Saved output file
    '''
    # Create a workbook for saving data
    workbook = xlsxwriter.Workbook(workbook_out)

    # to hold contour names and image set names
    contour_names = []
    
    # define initial variables
    file_names = [fname for fname in os.listdir(ct_folders) if not fname.startswith('.')]

    # go through all images in data base
    for MRN in file_names:
        print(MRN)
        rts_folder = os.path.join(ct_folders, MRN, 'RTS/')
        rts_files = glob.glob(os.path.join(rts_folder,'*.dcm'), recursive=None)
        rts_dicts = []

        # get all data from rtss files and put in an array 
        for file_path in rts_files:
            contours, contour_name, image_set = convert_to_dict(file_path)
            contour_dict, zunique, indexmin, indexmax = get_contour_points_dict(contours, structures_wanted)
            rts_dicts.append((contour_dict, zunique, indexmin, indexmax))
            contour_names.append(str(contour_name) + ", " + str(image_set))

        msd1, rms1, hd1, dsc1 = calc_similarity_metrics(rts_dicts[0][0], rts_dicts[0][1], rts_dicts[0][2], rts_dicts[0][3], rts_dicts[1][0], rts_dicts[1][1], rts_dicts[1][2], rts_dicts[1][3])
        msd2, rms2, hd2, dsc2 = calc_similarity_metrics(rts_dicts[0][0], rts_dicts[0][1], rts_dicts[0][2], rts_dicts[0][3], rts_dicts[2][0], rts_dicts[2][1], rts_dicts[2][2], rts_dicts[2][3])
        msd3, rms3, hd3, dsc3 = calc_similarity_metrics(rts_dicts[1][0], rts_dicts[1][1], rts_dicts[1][2], rts_dicts[1][3], rts_dicts[2][0], rts_dicts[2][1], rts_dicts[2][2], rts_dicts[2][3])


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

def fill_contour_dict(MRN, ct_folders, ROI_contour_dict, structures_wanted) : 
    '''
    fill_contour_dict fills a dictionary with the contour information for the structure wanted (prostate) with
    the autocontoured prostate for the TPCT and the Fused images of a single patient. 

    Parameters
    ----------
    MRN : String
        medical reference number of patient
    ct_folders : String
        path to folder containing patients CT image and rts files
    ROI_contour_dict : nested dictionary 
        Passed in empty with MRN initialized (metadata_dict = {'MRN' = {}})
    structures_wanted : list
        List of strings, with possible names for region of interest to be examined
        Eg ["PROSTATE", "prostate", "Prostate"]
    Returns
    -------
    ROI_contour_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED'
        Third level of dictionary hold the variables with keys : 'contour', 'zunique', 
            'indexmin', 'indexmax', 'volume', 'COM'
    '''
    rts_folder = os.path.join(ct_folders, MRN, 'RTS/')
    rts_files = glob.glob(os.path.join(rts_folder,'*.dcm'), recursive=None)

    for file_path in rts_files:
        contours, contour_name, image_set = convert_to_dict(file_path)
        #contour_dict, zunique, indexmin, indexmax = get_contour_points_dict(contours, structures_wanted)
        # select only for autocontoured ROIs
        if 'Elekta' in contour_name : 
            # check if TPCT image
            if any(substring in image_set for substring in ['PROS', 'PELVIS','PELVS', 'NO CATH', 'WITHOUT CATH']) :
                contour_dict_TPCT, zunique_TPCT, indexmin_TPCT, indexmax_TPCT = get_contour_points_dict(contours, structures_wanted)
                volume_TPCT, center_of_mass_TPCT = get_volume_and_center_ROI(contour_dict_TPCT, zunique_TPCT, indexmin_TPCT, indexmax_TPCT)
                # fill ROI_contour_dict
                ROI_contour_dict[MRN]['TPCT'] = {'contour' : contour_dict_TPCT, 'zunique' : zunique_TPCT, 'indexmin' : indexmin_TPCT, 
                                                 'indexmax' : indexmax_TPCT, 'volume' : volume_TPCT, 'COM' : center_of_mass_TPCT}
            # check if Fused image
            elif 'Fused' in image_set : 
                contour_dict_Fused, zunique_Fused, indexmin_Fused, indexmax_Fused = get_contour_points_dict(contours, structures_wanted)
                volume_Fused, center_of_mass_Fused = get_volume_and_center_ROI(contour_dict_Fused, zunique_Fused, indexmin_Fused, indexmax_Fused)
                # fill ROI_contour_dict
                ROI_contour_dict[MRN]['FUSED'] = {'contour' : contour_dict_Fused, 'zunique' : zunique_Fused, 'indexmin' : indexmin_Fused, 
                                                  'indexmax' : indexmax_Fused, 'volume' : volume_Fused, 'COM' : center_of_mass_Fused}
            # if neither, indicate warning
            else : 
                print("Warning! Incorrect image set names in patient,", MRN)
                print('Contour name : ', contour_name, "\nImage name : ", image_set)

    return ROI_contour_dict 

def get_fiducial_change(MRN, fiducial_dict, ROI_contour_dict, structures_wanted) : 
    '''
    get_fiducial_change gets the positions of the fiducials from the filled fiducial_dict 
    and determines how the positions change. It finds the average distance from the fiducials 
    to te center of mass of the fiducials, the avaerage distance from the fiducials to the 
    center of mass of the ROI, and the average interfiducials distance. It saved these for the 
    TPCT and Fused Images in the fiducial_dict, as well as introducing another sub-dictionary within 
    fiducial_dict that keeps track of the changes in these values. 

    Parameters
    ----------
    MRN : String
        medical reference number of patient
    fiducial_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED' and 'CHANGE'
        Third level of dictionary hold the variables with keys : 'locations', 'COM'
    ROI_contour_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED'
        Third level of dictionary hold the variables with keys : 'contour', 'zunique', 
            'indexmin', 'indexmax', 'volume', 'COM'
    structures_wanted : list
        List of strings, with possible names for region of interest to be examined
        Eg ["PROSTATE", "prostate", "Prostate"]
    Returns
    -------
    fiducial_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED' and 'CHANGE'
        Third level of dictionary hold the variables with keys : 'locations', 'COM'
            'avg fid to fid COM', 'avg fid to ROI COM', 'avg interfiducial distance'
        The following key are added to 'CHANGE' : 'locations', 'avg fid to fid COM'
            'avg fid to ROI COM', 'avg interfiducial distance'
    '''
    # get data for current patient, easier to read code
    center_of_mass_TPCT = fiducial_dict[MRN]['TPCT']['COM']
    center_of_mass_Fused = fiducial_dict[MRN]['FUSED']['COM']
    fiducials_TPCT = fiducial_dict[MRN]['TPCT']['locations']
    fiducials_Fused = fiducial_dict[MRN]['FUSED']['locations']
    center_of_ROI_TPCT = ROI_contour_dict[MRN]['TPCT']['COM']
    center_of_ROI_Fused = ROI_contour_dict[MRN]['FUSED']['COM']

    # determine average distance from fiducials to fiducial COM
    avg_dist_fidCOM_TPCT = average_fid_distance(fiducials_TPCT, center_of_mass_TPCT)
    avg_dist_fidCOM_Fused = average_fid_distance(fiducials_Fused, center_of_mass_Fused)
    # determine change in average distance from fiducials to fiducial COM 
    change_dist_fidCOM = avg_dist_fidCOM_Fused - avg_dist_fidCOM_TPCT

    # determine average distance from fiducials to ROI COM
    avg_dist_ROI_COM_TPCT = average_fid_distance(fiducials_TPCT, center_of_ROI_TPCT)
    avg_dist_ROI_COM_Fused = average_fid_distance(fiducials_Fused, center_of_ROI_Fused)
    # determine change in average distance from fiducials to ROI COM 
    change_dist_ROI_COM = avg_dist_ROI_COM_Fused - avg_dist_ROI_COM_TPCT

    # determine change in interfiducial distances
    interfiducial_dist_TPCT = average_interfiducial_distance(fiducials_TPCT)
    interfiducial_dist_Fused = average_interfiducial_distance(fiducials_Fused)
    change_in_interfiducial_dist = interfiducial_dist_Fused - interfiducial_dist_TPCT

    # determine change in each fiducial position
    position_change = change_fid_positions(fiducials_TPCT, fiducials_Fused)

    # add new feilds to dictionaries 
    # avareage distances to fid and ROI COM
    fiducial_dict[MRN]['TPCT']['avg fid to fid COM'] = avg_dist_fidCOM_TPCT
    fiducial_dict[MRN]['TPCT']['avg fid to ROI COM'] = avg_dist_ROI_COM_TPCT
    fiducial_dict[MRN]['TPCT']['avg interfiducial distance'] = interfiducial_dist_TPCT
    fiducial_dict[MRN]['FUSED']['avg fid to fid COM'] = avg_dist_fidCOM_Fused
    fiducial_dict[MRN]['FUSED']['avg fid to ROI COM'] = avg_dist_ROI_COM_Fused
    fiducial_dict[MRN]['FUSED']['avg interfiducial distance'] = interfiducial_dist_Fused
    # changes in fiducial locations 
    fiducial_dict[MRN]['CHANGE'] = {'locations' : position_change, 'avg fid to fid COM' : change_dist_fidCOM, 
                                    'avg fid to ROI COM' : change_dist_ROI_COM, 'avg interfiducial distance' : change_in_interfiducial_dist}

    return fiducial_dict

def scale_contour_iso(MRN, metadata_dict, fiducial_dict, ROI_contour_dict) : 
    '''
    scale_contour_iso performs an isotropic scaling of the Fused (Catheterized) protate contour.
    It either grows or shrinks the Fused prostate contour isotropically based around the center
    of the prostate. To expand/contract isotropically, a scaling factor is determined 
    based on the change in the distance from the center of the prostate to the 
    fiducials. The scaling factor (C) is determined to be
        (r1 + r2 + ... + rN) / N = (Cr1' + Cr2' + ... + CrN') / N
        C = (r1 + r2 + ... + rN) / (r1' + r2' + ... + rN')
        C = average_distance_1 / average_distance_2
    where r1,r2,... is the distances from the center of the protate to the fiducials
    in image 1, and r1',r2;,... is the distances from the center of the protate to the fiducials
    in image 2, and N is the number of fiducials. 
    It also saves the scaled contour and associated parameters in the ROI_contour_dict under FUSED_ISO
    for further analysis. 

    Parameters
    ----------
    MRN : String
        medical reference number of patient
    metatdata_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED'
        Third level of dictionary hold the variables with keys : 'image shape', 'origin', 'pixel spacing'
    fiducial_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED' and 'CHANGE'
        Third level of dictionary for 'TPCT' and 'FUSED' hold the variables with keys : 
            'locations', 'COM', 'avg fid to fid COM', 'avg fid to ROI COM', 'avg interfiducial distance'
        Third level of dictionary for 'CHANGE' holds the variables with keys : 
            'locations', 'avg fid to fid COM', 'avg fid to ROI COM', 'avg interfiducial distance'
    ROI_contour_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED'
        Third level of dictionary hold the variables with keys : 'contour', 'zunique', 
            'indexmin', 'indexmax', 'volume', 'COM'

    Returns
    -------
    ROI_contour_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT', 'FUSED', and 'SCALED_ISO'
        Third level of dictionary hold the variables with keys : 'contour', 'zunique', 
            'indexmin', 'indexmax', 'volume', 'COM'
            'SCALED_ISO' also has key 'scaling factor'
    '''

    # determine scaling factor 
    avg_dist_ROI_COM_TPCT = fiducial_dict[MRN]['TPCT']['avg fid to ROI COM']
    avg_dist_ROI_COM_Fused = fiducial_dict[MRN]['FUSED']['avg fid to ROI COM']
    scaling_factor = get_iso_scale_factor(avg_dist_ROI_COM_TPCT, avg_dist_ROI_COM_Fused)

    # scale contour
    contour_dict = ROI_contour_dict[MRN]['FUSED']['contour']
    ROI_center = ROI_contour_dict[MRN]['FUSED']['COM']
    contour_dict_scaled, zunique_scaled, indexmin_scaled, indexmax_scaled = scale_dict(MRN, metadata_dict, contour_dict, ROI_center, scaling_factor)
    volume_scaled, center_of_mass_scaled = get_volume_and_center_ROI(contour_dict_scaled, zunique_scaled, indexmin_scaled, indexmax_scaled)

    # add scaled contour to dictionary 
    ROI_contour_dict[MRN]['FUSED_ISO'] = {'contour' : contour_dict_scaled, 'zunique' : zunique_scaled, 'indexmin' : indexmin_scaled, 
                                        'indexmax' : indexmax_scaled, 'volume' : volume_scaled, 'COM' : center_of_mass_scaled, 'scaling factor' : scaling_factor}
    

    return ROI_contour_dict

def compare_contours(MRN, ROI_contour_dict, contour_comparison_dict, contour_name_1, contour_name_2) : 
    '''
    compare_contours compares two contours form the ROI_contour_dict (contour_name_1 and contour_name_2).
    The comparision finds the mean surface distance, residual mean-square error, hausdorff distance, 
    and dice coefficient for the two surfaces. 
    The comparisons are then saved to the contour_comparison_dict. 

    Parameters
    ----------
    MRN : String
        medical reference number of patient
    ROI_contour_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT', 'FUSED', and 'SCALED_ISO'
        Third level of dictionary hold the variables with keys : 'contour', 'zunique', 
            'indexmin', 'indexmax', 'volume', 'COM'
            'SCALED_ISO' also has key 'scaling factor'
    contour_comparison_dict : nested dictionary 
        Passed in empty with MRN initialized (metadata_dict = {'MRN' = {}})
    contour_name_1 : String
        Name of truth contour for comparison, usually 'TPCT'
    contour_name_2 : String 
        Name of second contour for comparison, usually 'FUSED' or 'FUSED_ISO'

    Returns
    -------
    contour_comparison_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'contour_name_1_To_contour_name_2'
        Third level of dictionary hold the variables with keys : 
            'mean surface distance', 'residual mean-square error', 'hausdorff distance', 'dice coefficient'
    '''

    contour_dict_1 = ROI_contour_dict[MRN][contour_name_1]['contour']
    zunique_1 = ROI_contour_dict[MRN][contour_name_1]['zunique']
    indexmin_1 = ROI_contour_dict[MRN][contour_name_1]['indexmin']
    indexmax_1 = ROI_contour_dict[MRN][contour_name_1]['indexmax']
    contour_dict_2 = ROI_contour_dict[MRN][contour_name_2]['contour']
    zunique_2 = ROI_contour_dict[MRN][contour_name_2]['zunique']
    indexmin_2 = ROI_contour_dict[MRN][contour_name_2]['indexmin']
    indexmax_2 = ROI_contour_dict[MRN][contour_name_2]['indexmax']

    msd, rms, hd, dsc = calc_similarity_metrics(contour_dict_1, zunique_1, indexmin_1, indexmax_1, contour_dict_2, zunique_2, indexmin_2, indexmax_2)

    comparison_string = contour_name_1 + '_to_' + contour_name_2
    contour_comparison_dict[MRN][comparison_string] = {'mean surface distance' : msd, 'residual mean-square error' : rms, 
                                                       'hausdorff distance' : hd, 'dice coefficient' : dsc}

    return contour_comparison_dict

def get_bladder_retum_volumes(MRN, ct_folders, bladder_volume, rectum_volume) : 
    '''
    get_bladder_retum_volumes retreive the volmes of the bladder and the recturm for the TPCT and Fused Images, 
    as well as the change in volume for a patient. This is then saved in the dictionaries bladder_volume and rectum_volume.
    
    Parameters
    ----------
    MRN : String
        medical reference number of patient
    ct_folders : String
        path to folder containing patients CT image and rts files
    bladder_volume : nested dictionary 
        Passed in empty or with other MRN values initialised
    rectum_volume : nested dictionary 
        Passed in empty or with other MRN values initialised
    
    Returns
    -------
    bladder_volume : nested dictionary 
        New subdictionary added with current MRN
        bladder_volume[MRN] = {'TPCT' : volume, 'FUSED' : volume, 'CHANGE' : volume}
    rectum_volume : nested dictionary 
        New subdictionary added with current MRN
        rectum_volume[MRN] = {'TPCT' : volume, 'FUSED' : volume, 'CHANGE' : volume}
    '''


    bladder_names = ['BLADDER', 'Bladder', 'bladder']
    rectum_names = ['RECTUM', 'Rectum', 'rectum']

    rts_folder = os.path.join(ct_folders, MRN, 'RTS/')
    rts_files = glob.glob(os.path.join(rts_folder,'*.dcm'), recursive=None)

    bladder_volume[MRN] = {}
    rectum_volume[MRN] = {}

    for file_path in rts_files:
        contours, contour_name, image_set = convert_to_dict(file_path)
        contour_dict_bladder, zunique_bladder, indexmin_bladder, indexmax_bladder = get_contour_points_dict(contours, bladder_names)
        contour_dict_rectum, zunique_rectum, indexmin_rectum, indexmax_rectum = get_contour_points_dict(contours, rectum_names)
        # select only for autocontoured ROIs
        if 'Elekta' in contour_name : 
            # check if TPCT image
            if any(substring in image_set for substring in ['PROS', 'PELVIS', 'PELVS', 'NO CATH', 'WITHOUT CATH']) :
                volume_bladder, center_of_mass_bladder = get_volume_and_center_ROI(contour_dict_bladder, zunique_bladder, indexmin_bladder, indexmax_bladder)
                volume_rectum, center_of_mass_rectum = get_volume_and_center_ROI(contour_dict_rectum, zunique_rectum, indexmin_rectum, indexmax_rectum)
                # fill ROI_contour_dict
                bladder_volume[MRN]['TPCT'] = volume_bladder
                rectum_volume[MRN]['TPCT'] = volume_rectum
            # check if Fused image
            elif 'Fused' in image_set : 
                volume_bladder, center_of_mass_bladder = get_volume_and_center_ROI(contour_dict_bladder, zunique_bladder, indexmin_bladder, indexmax_bladder)
                volume_rectum, center_of_mass_rectum = get_volume_and_center_ROI(contour_dict_rectum, zunique_rectum, indexmin_rectum, indexmax_rectum)
                # fill ROI_contour_dict
                bladder_volume[MRN]['FUSED'] = volume_bladder
                rectum_volume[MRN]['FUSED'] = volume_rectum
            # if neither, indicate warning
            else : 
                print("Warning! Incorrect image set names in patient,", MRN, "for bladder and rectum.")
                print('Contour name : ', contour_name, "\nImage name : ", image_set)

    bladder_volume[MRN]['CHANGE'] = bladder_volume[MRN]['FUSED'] - bladder_volume[MRN]['TPCT']
    rectum_volume[MRN]['CHANGE'] = rectum_volume[MRN]['FUSED'] - rectum_volume[MRN]['TPCT']

    return bladder_volume, rectum_volume

def check_scaling(MRN, fiducial_dict, ROI_contour_dict) :
    '''
    check_scaling reads the values of the scaling factor, average fiducial to ROI center of mass distances,
    fiducial locations, and change in interfiducial distances. These values are printed to the console 
    and are returned to be appended to a list and saved to csv. 
    These values can be helpful to reivew to ensure the scaling is occuring as expected.

    Parameters
    ----------
    MRN : String
        medical reference number of patient
    fiducial_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT' and 'FUSED' and 'CHANGE'
        Third level of dictionary for 'TPCT' and 'FUSED' hold the variables with keys : 
            'locations', 'COM', 'avg fid to fid COM', 'avg fid to ROI COM', 'avg interfiducial distance'
        Third level of dictionary for 'CHANGE' holds the variables with keys : 
            'locations', 'avg fid to fid COM', 'avg fid to ROI COM', 'avg interfiducial distance'
    ROI_contour_dict : nested dictionary 
        First level of dictionary keys : MRN
        Second level of dictionary keys image type : 'TPCT', 'FUSED', and 'SCALED_ISO'
        Third level of dictionary hold the variables with keys : 'contour', 'zunique', 
            'indexmin', 'indexmax', 'volume', 'COM'
            'SCALED_ISO' also has key 'scaling factor'
    
    Returns
    -------
    out_dict : dictionary
        dictionary to be appended to list for saving to csv
        contaings keys : 'MRN', 'Scaling factor', 'Average fiducial distance to ROI COM TPCT',
                'Average fiducial distance to ROI COM Fused', 'Average fiducial distance to ROI COM Scaled',
                'Average change in interfiducial distances', 'Change in volume'
    '''
    scaler = ROI_contour_dict[MRN]['FUSED_ISO']['scaling factor']
    center = ROI_contour_dict[MRN]['FUSED']['COM']

    fiducial_dist_TPCT = fiducial_dict[MRN]['TPCT']['avg fid to ROI COM']
    fiducial_dist_Fused = fiducial_dict[MRN]['FUSED']['avg fid to ROI COM']
    fiducial_pos_Fused = fiducial_dict[MRN]['FUSED']['locations']
    interfid_change = fiducial_dict[MRN]['CHANGE']['avg interfiducial distance']
    fiducial_pos_Scaled = []
    for point in fiducial_pos_Fused : 
        fiducial_pos_Scaled.append(scale_point(point, center, scaler))

    fiducial_dist_Scaled = average_fid_distance(fiducial_pos_Scaled, center)
    print("Check scaling : ")
    print("Scaling factor", scaler)
    print("Average change in interfiducial distances :", interfid_change)
    print("Change in volume :", (ROI_contour_dict[MRN]['FUSED']['volume'] - ROI_contour_dict[MRN]['TPCT']['volume']))

    out_dict = {'MRN' : MRN,
                'Scaling factor' : scaler,
                'Average fiducial distance to ROI COM TPCT' : fiducial_dist_TPCT,
                'Average fiducial distance to ROI COM Fused' : fiducial_dist_Fused,
                'Average fiducial distance to ROI COM Scaled' : fiducial_dist_Scaled,
                'Average change in interfiducial distances' : interfid_change,
                'Change in volume' : (ROI_contour_dict[MRN]['FUSED']['volume'] - ROI_contour_dict[MRN]['TPCT']['volume'])}

    return out_dict

def testing():

    # files and folders 
    ct_folders = r'/Volumes/External Drive/Redo/CT_files/'
    csv_reference_locations = r'/Volumes/External Drive/Redo/fiducial_reference_locations.csv'
    csv_metadata = r'/Volumes/External Drive/Redo/metadata.csv'
    csv_fiducial = r'/Volumes/External Drive/Redo/fiducial.csv'
    csv_comparison = r'/Volumes/External Drive/Redo/comparison.csv'
    workbook_out = r'/Volumes/External Drive/Redo/contour_data.xlsx'
    plot_folder = r'/Volumes/External Drive/Redo/Plots'
    check_scaling_csv = r'/Volumes/External Drive/Redo/check_scaling.csv'

    # constants 
    structures_wanted = ['Prostate', 'prostate', 'PROSTATE', 'PROS1_3625']

    # initialize data structures
    metadata_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}}}
    fiducial_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}, 'CHANGE' : {...}}}
    ROI_contour_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}, 'FUSED_ISO' : {...}}}
    contour_comparison_dict = {} # {MRN : {'TPCT_to_FUSED' : {...}, 'TPCT_to_FUSED_ISO' : {...}}}
    bladder_volume = {} # {MRN : {'TPCT' : volume, 'FUSED' : volume, 'CHANGE' : volume}}
    rectum_volume = {} # {MRN : {'TPCT' : volume, 'FUSED' : volume, 'CHANGE' : volume}}
    scaling_list = []


    # get DICOM file metadata and get fiducial data 
    metadata_dict, fiducial_dict = extract_data(csv_reference_locations, csv_metadata, csv_fiducial, ct_folders, metadata_dict, fiducial_dict)
    
    # read in DICOM file metadata and get fiducial data if already extracted 
    #metadata_dict = read_metadata_location_csv(csv_metadata, metadata_dict)
    #fiducial_dict = read_fiducial_location_csv(csv_fiducial, fiducial_dict)

    # review contour comparisons, creates workbook with msd, rms, hd, and dsc, optional
    #review_contour_data(ct_folders, workbook_out, structures_wanted)
    
    # get contour data, and analysis 
    print("Starting contour comparisons")
    for MRN in fiducial_dict.keys() : 
        print(MRN)
        ROI_contour_dict[MRN] = {}
        contour_comparison_dict[MRN] = {}
        # get contour data, and ROI center
        ROI_contour_dict = fill_contour_dict(MRN, ct_folders, ROI_contour_dict, structures_wanted)
        # get fiducial change data 
        fiducial_dict = get_fiducial_change(MRN, fiducial_dict, ROI_contour_dict, structures_wanted)
        # get isotropically scaled ROI
        ROI_contour_dict = scale_contour_iso(MRN, metadata_dict, fiducial_dict, ROI_contour_dict)
        # compare contours 
        contour_comparison_dict = compare_contours(MRN, ROI_contour_dict, contour_comparison_dict, 'TPCT', 'FUSED')
        contour_comparison_dict = compare_contours(MRN, ROI_contour_dict, contour_comparison_dict, 'TPCT', 'FUSED_ISO')
        # get bladder and rectum volume for plotting
        bladder_volume, rectum_volume = get_bladder_retum_volumes(MRN, ct_folders, bladder_volume, rectum_volume)
        # check scaling
        scaling_list.append(check_scaling(MRN, fiducial_dict, ROI_contour_dict))

        #print("protate volume TPCT:", ROI_contour_dict[MRN]['TPCT']['volume'])
        #print("protate volume Fused:", ROI_contour_dict[MRN]['FUSED']['volume'])


    print("Saving data")
    write_csv_comparison(csv_comparison, fiducial_dict, contour_comparison_dict)
    plotting(fiducial_dict, ROI_contour_dict, contour_comparison_dict, bladder_volume, rectum_volume, plot_folder)
    write_csv_from_dict(check_scaling_csv, scaling_list)

    return 

def main():
    '''
    The main function in this package runs the analysis of the prostate data in the 
    associated database. To run the program enter into the command line 
    $ python protate_analysis.py

    Before running the program, you must make a few modification to the main function
    in order to determine what analysis to run. 

    Prerequsites for running : 
    1. You must have python 3 installed
    2. You must intall the packages : numpy, scipy, matplotlib, pydicom, openCV (cv2), astropy
    3. You must have extracted the fiducial location form the treatment logfile (see function_fileIO.py main function)
    4. You must decide if you wish to extract the fiducial data. This must be done if the program has 
        not previously been run, and hte files metadata.csv, and fiducial.csv must be created and in the proper folders.
        You can always re-run the data extraction, but it is not required, and is relatively slow. 
        To extract data set EXTRACT_DATA = True, otherwise set EXTRACT_DATA = False
    5. You must select if you wish to review the contour extraction, comparisons, and scaling. 
        This will save the data to an excel file and csv, named contour_data.xlsx and check_scaling.csv, respectively. 
        It is not required for any of the analysis, but can provide some extra data. 
        To review data set REVIEW = True, otherwise set REVIEW = False.
    
    This function performs the following analysis tasks : 
    1. Extracts the fiducial positions and relevent metada from the DICOM files. 
        The fiducial search locations are determined by the reference locations in the 
        treatment logfiles, which must have already been extracted and saved to fiducial_reference_locations.csv.
        The fiducial location is found by searching in a radius around the reference location, 
        and performing a thresholding on this area of the image. The center of mass (or brightness)
        is then set as the fiducial location. This extracted data is then saved in a csv files 
        named metadata.csv and fiducial.csv. The data is also stored in dictionaries for further analysis. 
    2. Extracts the contours from the structure files of the CT images. 
        The contour points of all the structures is read in and saved to a dictionary. 
        this is done for each contour, the munal contour, and both autocontours. 
        The auto-contoured contour of interest, or protate, is then read in and saved to 
        a dictionary. For the non-catheterized/treatment planing CT (TPCT), and the catheterized/fused image,
        the contour points, a set of the axial slice location containing the contour, 
        the minimum and maxmimum indeces of the smallest bounding box of the contour, 
        the volume of the contour, and the center location of the contour are all saved 
        to the dictionary for futher analysis. 
    3. Determine changes in fiducials between images. 
        Determine the cahnge in the average distances from the fiducials to 
        the center of mass of the all the fiducials, the average distance from the 
        fiducials to the center of the ROI (protate), the average interfidcial distance,
        and the changes in all of these values between the catheterized and non-catheterized images.
    4. Perform an isotropic scaling of the Fused (Catheterized) protate contour.
        Either grow or shrink the Fused prostate contour isotropically based around the center
        of the prostate. To expand/contract isotropically, a scaling factor is determined 
        based on the change in the distance from the center of the prostate to the 
        fiducials. The scaling factor (C) is determined to be
            (r1 + r2 + ... + rN) / N = (Cr1' + Cr2' + ... + CrN') / N
            C = (r1 + r2 + ... + rN) / (r1' + r2' + ... + rN')
            C = average_distance_1 / average_distance_2
        where r1,r2,... is the distances from the center of the protate to the fiducials
        in image 1, and r1',r2;,... is the distances from the center of the protate to the fiducials
        in image 2, and N is the number of fiducials. 
    5. The contours for the scaled and unscaled catheterized CT images (Fused) are compared to the non-catheterized image (TPCT).
        The mean surface distance, the residual mean-square error, the hausdorff distance, 
        and the dice coefficient are determined between the non-catheterized image (TPCT) 
        and the unscaled cathterised image (Fused), and between the non-catheterized image (TPCT) 
        and the isotripically scaled cathterised image.
    6. The bladder and rectum volumes are extracted for plotting.
    7. Data is plotted and saved. 
    '''
    # set this value to true, if you wish to extract and save the data 
    EXTRACT_DATA = True 
    REVIEW = False 

    # files and folders 
    ct_folders = r'/Volumes/External Drive/ProstateDBWithContours/CT_files/'
    csv_reference_locations = r'/Volumes/External Drive/ProstateDBWithContours/fiducial_reference_locations.csv'
    csv_metadata = r'/Volumes/External Drive/ProstateDBWithContours/metadata.csv'
    csv_fiducial = r'/Volumes/External Drive/ProstateDBWithContours/fiducial.csv'
    csv_comparison = r'/Volumes/External Drive/ProstateDBWithContours/comparison.csv'
    workbook_out = r'/Volumes/External Drive/ProstateDBWithContours/contour_data.xlsx'
    plot_folder = r'/Volumes/External Drive/ProstateDBWithContours/Plots'
    check_scaling_csv = r'/Volumes/External Drive/ProstateDBWithContours/check_scaling.csv'

    # constants 
    structures_wanted = ['Prostate', 'prostate', 'PROSTATE', 'PROS1_3625', 'GTV', 'CTV', 'CTV1', 'CTV Prostate']

    # initialize data structures
    metadata_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}}}
    fiducial_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}, 'CHANGE' : {...}}}
    ROI_contour_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}, 'FUSED_ISO' : {...}}}
    contour_comparison_dict = {} # {MRN : {'TPCT_to_FUSED' : {...}, 'TPCT_to_FUSED_ISO' : {...}}}
    bladder_volume = {} # {MRN : {'TPCT' : volume, 'FUSED' : volume, 'CHANGE' : volume}}
    rectum_volume = {} # {MRN : {'TPCT' : volume, 'FUSED' : volume, 'CHANGE' : volume}}
    scaling_list = []

    if EXTRACT_DATA :
        # get DICOM file metadata and get fiducial data 
        metadata_dict, fiducial_dict = extract_data(csv_reference_locations, csv_metadata, csv_fiducial, ct_folders, metadata_dict, fiducial_dict)
    else :    
        # read in DICOM file metadata and get fiducial data if already extracted 
        metadata_dict = read_metadata_location_csv(csv_metadata, metadata_dict)
        fiducial_dict = read_fiducial_location_csv(csv_fiducial, fiducial_dict)

    if REVIEW_CONTOUR :
        # review contour comparisons, creates workbook with msd, rms, hd, and dsc, optional
        print("Reviewing contours : ")
        review_contour_data(ct_folders, workbook_out, structures_wanted)
    
    # get contour data, and analysis 
    print("Starting contour comparisons")
    for MRN in fiducial_dict.keys() : 
        print("Contour analysis :", MRN)
        ROI_contour_dict[MRN] = {}
        contour_comparison_dict[MRN] = {}
        # get contour data, and ROI center
        ROI_contour_dict = fill_contour_dict(MRN, ct_folders, ROI_contour_dict, structures_wanted)
        # get fiducial change data 
        fiducial_dict = get_fiducial_change(MRN, fiducial_dict, ROI_contour_dict, structures_wanted)
        # get isotropically scaled ROI
        ROI_contour_dict = scale_contour_iso(MRN, metadata_dict, fiducial_dict, ROI_contour_dict)
        # compare contours 
        contour_comparison_dict = compare_contours(MRN, ROI_contour_dict, contour_comparison_dict, 'TPCT', 'FUSED')
        contour_comparison_dict = compare_contours(MRN, ROI_contour_dict, contour_comparison_dict, 'TPCT', 'FUSED_ISO')
        # get bladder and rectum volume for plotting
        bladder_volume, rectum_volume = get_bladder_retum_volumes(MRN, ct_folders, bladder_volume, rectum_volume)
        if REVIEW : 
            # check scaling
            scaling_list.append(check_scaling(MRN, fiducial_dict, ROI_contour_dict))

    print("Saving data")
    
    write_csv_comparison(csv_comparison, fiducial_dict, contour_comparison_dict)
    plotting(fiducial_dict, ROI_contour_dict, contour_comparison_dict, bladder_volume, rectum_volume, plot_folder)
    
    if REVIEW : 
        write_csv_from_dict(check_scaling_csv, scaling_list)

    return 

if __name__ == "__main__":
    testing()
    pass