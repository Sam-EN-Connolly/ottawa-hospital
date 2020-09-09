import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


###################################################################
############################ XML files ############################
###################################################################

def read_plan_XML_file(file_path):
    '''
    read_plan_XML_file reads one of the plan XML files
    and extracts the MRN, thepositions of the reference fiducials in the TPCT.

    Parameters
    ----------
    file_path : string
        path to the XML file

    Returns
    -------
    MRN : int
        the MRN of the patient
    fiducial_locations : list (2d)
        list of locations of fiducials in TPCT from logfile. 
        the fiducial locations are given in lists of [z,y,x].
    '''
    # works for v.10.5x and v.9.5x
    tree = ET.parse(file_path)
    root = tree.getroot()  
    # extract basic information in each xml file:
    MRN = root.findall("./{http://www.accuray.com/cyris}PATIENT_PROFILE/{http://www.accuray.com/cyris}MEDICAL_ID")
    MRN = MRN[0].text

    # extract fiducial locations
    fiducial_locations = []
    fid_tag=root.findall("./{http://www.accuray.com/cyris}ALIGN_SETUP/{http://www.accuray.com/cyris}FIDUCIALSET/{http://www.accuray.com/cyris}FIDUCIAL")
    for fid in fid_tag : 
        x_pos = fid.find('{http://www.accuray.com/cyris}X').text
        y_pos = fid.find('{http://www.accuray.com/cyris}Y').text
        z_pos = fid.find('{http://www.accuray.com/cyris}Z').text
        fiducial_locations.append([float(z_pos), float(y_pos), float(x_pos)])

    return MRN, fiducial_locations


###################################################################
############################ CSV files ############################
###################################################################

def write_csv_from_dict(csv_file_path, dict_data) :
    '''
    write_csv_from_dict writes data stored as dictionaries in a list to a csv file.

    Parameters
    ----------
    csv_file_path : string
        path and name of csv file to write
    dict_data : list of dictionaries
        data to write to csv file. collumns will be labeled with the keys
        or each item in the dictionary. all dictionaries must have the same keys. 

    Returns
    -------
    None
    '''
    csv_columns = dict_data[0].keys()
    try:
        with open(csv_file_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)

    except IOError:
        print('I/O error')

    return None 

def read_reference_location_csv(csv_file_path) : 
    '''
    read_reference_location_csv reads the csv file containing the reference fiducial locations,
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
    fiducial_dict = {}
    with open(csv_file_path, 'r') as csv_file : 
        csv_reader = csv.DictReader(csv_file)
        for line in csv_reader : 
            fid_list = eval(line['Fiducial Locations [z,y,x]'])
            fiducial_dict[line['MRN']]  = fid_list
    return fiducial_dict

def read_metadata_location_csv(csv_metadata, metadata_dict) : 
    '''
    reads in metadata file into metadata_dict 

    Parameters
    ----------
    csv_metadata : string
        path and name of csv file to read
    metadata_dict : dict
        epmty dict to hold metadata

    Returns
    -------
    metadata_dict : dict
        dict with corectly formatted metadata
    '''

    with open(csv_metadata, 'r') as csv_file : 
        csv_reader = csv.DictReader(csv_file)
        for line in csv_reader : 
            MRN = str(line['MRN'])
            # add zero to begining if it has been removed when reading in file, all MRNs should be 8 digits
            if len(MRN) == 7 :
                MRN = '0' + MRN 
            img_shape_TPCT = eval(line['Image Shape TPCT'])
            origin_TPCT = eval(line['Origin TPCT'])
            pixel_spacing_TPCT = eval(line['Pixel Spacing TPCT'])
            img_shape_Fused = eval(line['Image Shape Fused'])
            origin_Fused = eval(line['Origin Fused'])
            pixel_spacing_Fused = eval(line['Pixel Spacing Fused'])

            metadata_dict[MRN] = {'TPCT' : {}, 'FUSED' : {}}
            metadata_dict[MRN]['TPCT'] = {'image shape' : img_shape_TPCT, 'origin' : origin_TPCT, 'pixel spacing' : pixel_spacing_TPCT}
            metadata_dict[MRN]['FUSED'] = {'image shape' : img_shape_Fused, 'origin' : origin_Fused, 'pixel spacing' : pixel_spacing_Fused}

    return metadata_dict

def read_fiducial_location_csv(csv_fiducial, fiducial_dict) : 
    '''
    reads in fiducial file into fiducial_dict 

    Parameters
    ----------
    csv_fiducial : string
        path and name of csv file to read
    metatdata_dict : dict
        epmty dict to hold metatdata

    Returns
    -------
    metatdata_dict : dict
        dict with corectly formatted fiducial data
    '''

    with open(csv_fiducial, 'r') as csv_file : 
        csv_reader = csv.DictReader(csv_file)
        for line in csv_reader : 
            MRN = str(line['MRN'])
            # add zero to begining if it has been removed when reading in file, all MRNs should be 8 digits
            if len(MRN) == 7 :
                MRN = '0' + MRN
            fiducial_points_found_TPCT_mm = np.array(eval(line['Fiducial Positions TPCT']))
            fiducialCM_TPCT_mm = np.array(eval(line['Fiducial COM TPCT']))
            fiducial_points_found_Fused_mm = np.array(eval(line['Fiducial Positions Fused']))
            fiducialCM_Fused_mm = np.array(eval(line['Fiducial COM Fused']))

            fiducial_dict[MRN] = {'TPCT' : {}, 'FUSED' : {}}
            fiducial_dict[MRN]['TPCT'] = {'locations' : fiducial_points_found_TPCT_mm, 'COM' : fiducialCM_TPCT_mm}
            fiducial_dict[MRN]['FUSED'] = {'locations' : fiducial_points_found_Fused_mm, 'COM' : fiducialCM_Fused_mm}

    return fiducial_dict

def write_csv_comparison(csv_comparison, fiducial_dict, contour_comparison_dict) :
    '''
    writes comparison data to csv files 

    Parameters
    ----------
    csv_comparison : string
        path and name of csv file to write
    fiducial_dict : dict
        dict holding fiducial posisions and changes
    contour_comparison_dict : dict
        dict holding comparisons between contours

    Returns
    -------
    None
    '''
    with open(csv_comparison, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['MRN', 
                         'Change in fiducial locations (TPCT to Fused)', 
                         'Average change in interfiducial distances (TPCT to Fused)',
                         'Change in average fiducial location from center of ROI (TPCT to Fused)',
                         'Hausdorff distance (no deformation)', 
                         'Dice Coefficient (no deformation)',
                         'Mean Surface Distance (no deformation)',
                         'Residual Mean-Square Error (no deformation)',
                         'Hausdorff distance (isotropic)', 
                         'Dice Coefficient (isotropic)',
                         'Mean Surface Distance (isotropic)',
                         'Residual Mean-Square Error (isotropic)'])
        for MRN in contour_comparison_dict.keys() : 
            out_list = [MRN, 
                        fiducial_dict[MRN]['CHANGE']['locations'], 
                        np.mean(fiducial_dict[MRN]['CHANGE']['locations']),
                        fiducial_dict[MRN]['CHANGE']['avg fid to ROI COM'],
                        contour_comparison_dict[MRN]['TPCT_to_FUSED']['hausdorff distance'], 
                        contour_comparison_dict[MRN]['TPCT_to_FUSED']['dice coefficient'],
                        contour_comparison_dict[MRN]['TPCT_to_FUSED']['mean surface distance'],
                        contour_comparison_dict[MRN]['TPCT_to_FUSED']['residual mean-square error'],
                        contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['hausdorff distance'], 
                        contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['dice coefficient'],
                        contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['mean surface distance'],
                        contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['residual mean-square error']]
            writer.writerow(out_list)

def plotting(fiducial_dict, ROI_contour_dict, contour_comparison_dict, bladder_volume, rectum_volume, plot_folder) : 

    MRN_vals = list(fiducial_dict.keys())
    MRN_vals = np.sort(MRN_vals)

    fid_to_ROI_change = []
    fid_to_fidCOM_change = []
    bladder_volume_change = []
    rectum_volume_change = []
    prostate_volume_change = []
    fid_position_change_avg = []
    #fid_position_change_std = []

    hd_unscaled = []
    dsc_unscaled = []
    msd_unscaled = []
    rms_unscaled = []

    #hd_scaled = []
    #dsc_scaled = []
    #msd_scaled = []
    #rms_scaled = []

    for MRN in MRN_vals : 
        fid_to_ROI_change.append(fiducial_dict[MRN]['CHANGE']['avg fid to ROI COM'])
        fid_to_fidCOM_change.append(fiducial_dict[MRN]['CHANGE']['avg fid to fid COM'])
        bladder_volume_change.append(bladder_volume[MRN]['CHANGE'])
        rectum_volume_change.append(rectum_volume[MRN]['CHANGE'])
        prostate_volume_change.append(ROI_contour_dict[MRN]['FUSED']['volume'] - ROI_contour_dict[MRN]['TPCT']['volume'])
        fid_position_change_avg.append(np.mean(fiducial_dict[MRN]['CHANGE']['avg interfiducial distance']))
        #fid_position_change_std.append(np.std(fiducial_dict[MRN]['CHANGE']['locations']))

        hd_unscaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED']['hausdorff distance'])
        dsc_unscaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED']['dice coefficient'])
        msd_unscaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED']['mean surface distance'])
        rms_unscaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED']['residual mean-square error'])

        hd_scaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['hausdorff distance'])
        dsc_scaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['dice coefficient'])
        msd_scaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['mean surface distance'])
        rms_scaled.append(contour_comparison_dict[MRN]['TPCT_to_FUSED_ISO']['residual mean-square error'])

    # plot change in prostate volume vs change in bladder volume
    #corelation_plot(bladder_volume_change, prostate_volume_change, "Bladder vs Prostate Volume Changes", "Change in Bladder Volume (mm^3)", "Change in Prostate Volume (mm^3)", plot_folder)
    # plot change in prostate volume vs change in rectum volume
    #corelation_plot(rectum_volume_change, prostate_volume_change, "Rectum vs Prostate Volume Changes", "Change in Rectum Volume (mm^3)", "Change in Prostate Volume (mm^3)", plot_folder)
   
    # plot change in prostate volume vs haudorf distance
    #corelation_plot(prostate_volume_change, hd_unscaled, "Hausdorff Distance vs Change in Prostate Volume", "Change in Prostate Volume (mm^3)", "Hausdorff Distance (mm)", plot_folder)
    # plot change in prostate volume vs dice coefficient
    #corelation_plot(prostate_volume_change, dsc_unscaled, "Dice Coefficient vs Change in Prostate Volume",  "Change in Prostate Volume (mm^3)", "Dice Coefficient", plot_folder)
    # plot change in prostate volume vs mean surface distance 
    #corelation_plot(prostate_volume_change, msd_unscaled, "Mean Surface Distance vs Change in Prostate Volume", "Change in Prostate Volume (mm^3)", "Mean Surface Distance (mm)", plot_folder)
    
    # plot change in average fiducial distance to fiducial center of mass vs haudorf distance
    corelation_plot(fid_to_fidCOM_change, hd_unscaled, "Hausdorff Distance vs Average Fiducial Distance to Fiducial Center of Mass", "Change in Average Fiducial Distance (mm)", "Hausdorff Distance (mm)", plot_folder)
    # plot change in average fiducial distance to fiducial center of mass vs dice coefficient
    corelation_plot(fid_to_fidCOM_change, dsc_unscaled, "Dice Coefficient vs Average Fiducial Distance to Fiducial Center of Mass", "Change in Average Fiducial Distance (mm)",  "Dice Coefficient", plot_folder)
    # plot change in average fiducial distance to fiducial center of mass vs mean surface distance 
    corelation_plot(fid_to_fidCOM_change, msd_unscaled, "Mean Surface Distance vs Average Fiducial Distance to Fiducial Center of Mass", "Change in Average Fiducial Distance (mm)",  "Mean Surface Distance (mm)", plot_folder)
    
    # plot change in average fiducial distance to ROI center of mass vs haudorf distance
    corelation_plot(fid_to_ROI_change, hd_unscaled, "Hausdorff Distance vs Average Fiducial Distance to ROI Center of Mass", "Change in Average Fiducial Distance (mm)", "Hausdorff Distance (mm)", plot_folder)
    # plot change in average fiducial distance to ROI center of mass vs dice coefficient
    corelation_plot(fid_to_ROI_change, dsc_unscaled, "Dice Coefficient vs Average Fiducial Distance to ROI Center of Mass", "Change in Average Fiducial Distance (mm)",  "Dice Coefficient", plot_folder)
    # plot change in average fiducial distance to ROI center of mass vs mean surface distance 
    corelation_plot(fid_to_ROI_change, msd_unscaled, "Mean Surface Distance vs Average Fiducial Distance to ROI Center of Mass", "Change in Average Fiducial Distance (mm)",  "Mean Surface Distance (mm)", plot_folder)
    
    # plot change in average change in fiducial spacing vs haudorf distance
    corelation_plot(fid_position_change_avg, hd_unscaled, "Hausdorff Distance vs Change in Average Interfiducial Distance", "Change in Average Interiducial Distance (mm)",  "Hausdorff Distance (mm)", plot_folder)
    # plot change in average change in fiducial spacing vs dice coefficient
    corelation_plot(fid_position_change_avg, dsc_unscaled, "Dice Coefficient vs Change in Average Interfiducial Distance", "Change in Average Interiducial Distance (mm)",  "Dice Coefficient", plot_folder)
    # plot change in average change in fiducial spacing vs mean surface distance 
    corelation_plot(fid_position_change_avg, msd_unscaled, "Mean Surface Distance vs Change in Average Interfiducial Distance", "Change in Average Interiducial Distance (mm)",  "Mean Surface Distance (mm)", plot_folder)

    # scatter plot of all MRN 
    corelation_plot_2(MRN_vals, hd_unscaled, hd_scaled, "Unscaled", "Isotopic Scaling", "Hausdorff Distance for Scaled and Unscaled Prostates", "MRN", "Hausdorff Distance (mm)", plot_folder)
    corelation_plot_2(MRN_vals, dsc_unscaled, dsc_scaled, "Unscaled", "Isotopic Scaling", "Dice Coefficient Distance for Scaled and Unscaled Prostates", "MRN", "Dice Coefficent", plot_folder)
    corelation_plot_2(MRN_vals, msd_unscaled, msd_scaled, "Unscaled", "Isotopic Scaling", "Mean Surface Distance Distance for Scaled and Unscaled Prostates", "MRN", "Mean Surface Distance (mm)", plot_folder)
    

def corelation_plot(x_data, y_data, title, x_label, y_label, plot_folder) :
    '''
    corelation_plot creates and saves a formatted scatter plot  
    ''' 
    plt.rcParams["figure.figsize"] = (20,15)

    fig, ax = plt.subplots()

    plt.scatter(x_data, y_data,  c='b', marker='x')

    # format axes and title
    ax.set(xlabel=x_label, ylabel=y_label)

    # add grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    #plt.show()

    plot_name = os.path.join(plot_folder, title + ".jpg")
    plt.savefig(plot_name)

def corelation_plot_2(x_data, y_data, y_data_2, name, name_2, title, x_label, y_label, plot_folder) :
    '''
    corelation_plot creates and saves a formatted scatter plot  
    ''' 
    fig, ax = plt.subplots()

    plt.scatter(x_data,y_data, c='b', marker='x', label=name)
    plt.scatter(x_data,y_data_2, c='r', marker='x', label=name_2)
    
    # format axes and title
    ax.set(xlabel=x_label, ylabel=y_label)

    # add grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend(loc='upper left')
    #plt.show()

    plot_name = os.path.join(plot_folder, title + ".jpg")
    plt.savefig(plot_name)

def main() : 
    '''
    Reads prostate plan xml files and saves to a csv file.
    '''
    csv_file_path = r"/Volumes/External Drive/ProstateDataBase/fiducial_reference_locations.csv"
    logfiles_folder = r"/Volumes/External Drive/ProstateDataBase/Logfiles/"

    dict_data = []
    for root, dirs, files in os.walk(logfiles_folder, topdown=False):
        for name in files : 
            if '.xml' in name : 
                xml_path = os.path.join(root, name)
                MRN, fiducial_locations = read_plan_XML_file(xml_path)
                dict_data.append({'MRN': MRN, 'Fiducial Locations [z,y,x]': fiducial_locations})
        print('Completed reading ' + xml_path)

    write_csv_from_dict(csv_file_path, dict_data)

    return 

if __name__ == "__main__":
    main()
    pass