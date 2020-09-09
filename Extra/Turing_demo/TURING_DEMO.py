
#///////////////////////////////////////////////////////////////////////////
#TO DO
#consider precomputed data for speed (ie images with contours on already)
#sagittal/coronal views (necessary?)
#option to change contour color
#put inputs into config file

# visual cue that new contours are done loading? 
#have to check to make sure relevant structures are in each dataset (either an if statement, or remove data with 
# missing structures prior to putting them into program)
#add option for with replacement/without replacement (might be interesting to see if same people consistently identify structures the same way)
#////////////////////////////////////////////////////////////////////////

# Notes
# - edited the admire contours by appending the first contour point as the last
#   because the manual structures have this to close the loop when plotting
# - reminder that warnings are suppressed due to smoothing
# - reminder that smoothing is on

import easygui
import sys
import os
import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import random
from scipy import interpolate
import warnings
import pickle
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.style.use('dark_background')

###############################################################################
# Prepare the CT and structure data
###############################################################################

def prepare_dicom_data_for_individual_case(rts_filename, list_of_ct_filenames, structure_wanted):
    
    # read in ct dicom slices
    ct_slices = []
    for ct_file in list_of_ct_filenames:
        ct_slices.append(dcm.dcmread(ct_file))
    # sort the 3d slice in axial order
    ct_slices = sorted(ct_slices, key=lambda s: s.SliceLocation)
    
    # pixel aspects, assuming all slices are the same, and rescale values
    ps = ct_slices[0].PixelSpacing
    ss = ct_slices[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    slope = ct_slices[0].RescaleSlope
    intercept = ct_slices[0].RescaleIntercept
    
    
    # create 3D array to hold CT image date
    img_shape = [len(ct_slices), 512, 512]
    img3d = np.zeros(img_shape)
    img_UIDs = []
    
    # read in structure dicom
    rts = dcm.dcmread(rts_filename)
    
    # find ROI number of contour of interest
    for structure in rts.StructureSetROISequence:
        if structure.ROIName == structure_wanted:
            structure_index = int(structure.ROINumber)
    
    # find ROIContourSequence index that holds contour of interest
    roi_contour_sequence_index = [int(x.ReferencedROINumber) for x in rts.ROIContourSequence].index(structure_index)
    
    # find list of CT slice SOP UIDs that have contour of interest on them
    SOP_UIDs = [x.ContourImageSequence[0].ReferencedSOPInstanceUID for x in rts.ROIContourSequence[roi_contour_sequence_index].ContourSequence]
    
    # fill 3D array with the images from the files
    # and get list of corresponding UIDs to pair w/ contours
    contour_matrix = []
    for i, s in enumerate(ct_slices):
        img2d = s.pixel_array
        img3d[i,:, :] = img2d 
        img_UIDs.append(s.SOPInstanceUID)
        
        #Get positional info for later plotting of structure contours
        x_origin = float(ct_slices[0].ImagePositionPatient[0])
        y_origin = float(ct_slices[0].ImagePositionPatient[1])
        pixel_size = float(ct_slices[0].PixelSpacing[0])
        
        #fill associated structure matrix
        if s.SOPInstanceUID in SOP_UIDs:
            
            contour_points_x = [[float(y) for y in x.ContourData[0::3]] for x in rts.ROIContourSequence[roi_contour_sequence_index].ContourSequence\
                                 if x.ContourImageSequence[0].ReferencedSOPInstanceUID == s.SOPInstanceUID ][0]
            contour_points_y = [[float(y) for y in x.ContourData[1::3]] for x in rts.ROIContourSequence[roi_contour_sequence_index].ContourSequence\
                                 if x.ContourImageSequence[0].ReferencedSOPInstanceUID == s.SOPInstanceUID ][0]
    
            #convert contour points y for orientaton (currently assumed HF supine!!!!)
            contour_points_y = [y_origin + img3d.shape[1]*pixel_size - (y-y_origin)  for y in contour_points_y]
            
            #append first point of contour to close structure if not already done (manual structures have this already)
            if contour_points_x[0] != contour_points_x[-1] or contour_points_y[0] != contour_points_y[-1]:
                contour_points_x.append(contour_points_x[0])
                contour_points_y.append(contour_points_y[0])
            
            contour_matrix.append([contour_points_x, contour_points_y])
            
            

        else:
            contour_matrix.append([np.nan, np.nan])
   
            
    #convert img3d pixel intensities to HU (for window/level presets later)
    img3d = img3d * slope + intercept

    return img3d, contour_matrix, x_origin, y_origin, pixel_size

    
def smooth_contour_matrix():
    global contour_matrix
    
    for i, contour in enumerate(contour_matrix):

        if np.nan not in contour:
            x=contour[0]
            y=contour[1]
            if len(x) >=4: #minimum of 4 points required for splprep

                #get rid of repeating points(not compatible with splprep)
                x_new=[]
                y_new=[]
                for k, (x_, y_) in enumerate(zip(x,y)):
                    if k>0:
                        if abs(x_ - x[k-1]) <0.1 and abs(y_ - y[k-1]<0.1):
                            pass
                        else:
                            x_new.append(x_)
                            y_new.append(y_)
                x=x_new
                y=y_new

                #interpolate
                tck,u = interpolate.splprep([x, y], s=10, per=1)
                unew = np.arange(0, 1.01, 0.01)
                out = interpolate.splev(unew, tck)
                contour_matrix[i][0] = out[0]
                contour_matrix[i][1] = out[1]



def get_random_structure_set():
    # randomly select a structure set to display then remove it from the list
#    rts_filename = random.choice(structure_list)
#    structure_list.remove(rts_filename)
    try:
        random_selection = random.choice([x for x in structure_list if x['structure'] == structure_wanted_human])
        rts_filename = random_selection['rts filename']
        structure_list.remove(random_selection)
    except IndexError:
        easygui.msgbox('User has evaluated the selected structures already.\nPlease restart and select a new structure.', 'Done')
        plt.close()

    

   
    # get ct data for selected structure_set, and determine proper structure_wanted name
    if machine_rts_filename in rts_filename:
        list_of_ct_filenames = [rts_filename.replace(machine_rts_filename, '') + x for x in os.listdir(rts_filename.replace(machine_rts_filename, '')) if 'CT' in x]
        structure_wanted = structure_wanted_machine
    if human_rts_filename in rts_filename:
        list_of_ct_filenames = [rts_filename.replace(human_rts_filename, '') + x for x in os.listdir(rts_filename.replace(human_rts_filename, '')) if 'CT' in x]
        structure_wanted = structure_wanted_human
        
    return structure_wanted, rts_filename, list_of_ct_filenames


###############################################################################       




###############################################################################
#   Handle GUI interactions
###############################################################################

def on_slider_change(val):
    global contour_plot
    #im.set_clim(vmin=-2000, vmax=-500)
    im.set_data(img3d[int(current_slice_slider.val), :, :])
    contour_plot[0].set_ydata(contour_matrix[int(current_slice_slider.val)][1])
    contour_plot[0].set_xdata(contour_matrix[int(current_slice_slider.val)][0])

    
def on_scroll(event):
    global current_slice_slider
    if event.button == 'up':
        if current_slice_slider.val < img3d.shape[0]:
            current_slice = current_slice_slider.val + 1
        else:
            current_slice = current_slice_slider.val
            
    else:
        if current_slice_slider.val > 0:
            current_slice = current_slice_slider.val - 1
        else:
            current_slice = current_slice_slider.val
    current_slice_slider.set_val(current_slice)


def on_machine_button_click(event):
    global rts_filename, img3d, contour_matrix, im, structure_wanted, list_of_ct_filenames, current_slice_slider,contour_plot
    
    if machine_rts_filename in rts_filename:
        print('Correct')
        user_data.append({'user': user, 'rts filename':rts_filename, 'structure':structure_wanted_human, 'machine/human':'machine', 'guess':'right'})
    else:
        print('Incorrect')
        user_data.append({'user': user, 'rts filename':rts_filename, 'structure':structure_wanted_human, 'machine/human':'human', 'guess':'wrong'})
    
    pickle.dump(user_data, open( results_folder + '//' + user + ".p", "wb" ) )

    if len([x for x in structure_list if x['structure'] == structure_wanted_human]) > 0:
        #generate new plot    
        generate_plots()
    else:
        plt.close()
        easygui.msgbox('There are no more of the selected structures in the dataset.\nPlease restart and select a new structure.', 'Done')  

def on_human_button_click(event):
    global rts_filename, img3d, contour_matrix, im, structure_wanted, list_of_ct_filenames, current_slice_slider,contour_plot
        
    if human_rts_filename in rts_filename:
        print('Correct')
        user_data.append({'user': user, 'rts filename':rts_filename, 'structure':structure_wanted_human, 'machine/human':'human', 'guess':'right'})
    else:
        print('Incorrect')
        user_data.append({'user': user, 'rts filename':rts_filename, 'structure':structure_wanted_human, 'machine/human':'machine', 'guess':'wrong'})
    
    pickle.dump(user_data, open( results_folder + '//' + user + ".p", "wb" ) )

    if len([x for x in structure_list if x['structure'] == structure_wanted_human]) > 0:
        #generate new plot    
        generate_plots()
    else:
        plt.close()
        easygui.msgbox('There are no more of the selected structures in the dataset.\nPlease restart and select a new structure.', 'Done')

    
def wl_update(label):
    global hu_min, hu_max
    if label == 'Abdomen':
        window = 400
        level = 50
    if label == 'Soft Tissue':
        window = 600
        level = 40
    if label == 'Head':
        window = 300
        level = 120
    if label == 'Brain':
        window = 200
        level = 70
    if label == 'Lung':
        window = 1700
        level = -300
    hu_min = level - window/2
    hu_max = level + window/2
    im.set_clim(vmin = hu_min, vmax = hu_max)
    fig.canvas.draw()
    fig.canvas.flush_events()
   

def generate_plots():
    global rts_filename, img3d, contour_matrix, im, structure_wanted, list_of_ct_filenames, current_slice_slider,contour_plot   

    # get all the relevant data (choose random structure set, get ct and contour data from dicoms)

    structure_wanted, rts_filename, list_of_ct_filenames = get_random_structure_set()

    img3d, contour_matrix, x_origin, y_origin, pixel_size = prepare_dicom_data_for_individual_case(rts_filename, list_of_ct_filenames, structure_wanted)

    #smooth the human contours if desired

    if smooth_human_contours == True and human_rts_filename in rts_filename:
        smooth_contour_matrix()
    
    # can remove ax.clear() for randomly generated colors of contours
    ax.clear()
    
    # deal with the Slider
    # try statement only runs if it's NOT the first image plotted (ie slider doesn't exist yet, needs to be initialized)
    # subsequent images just update the slider rather than re-creating it
    try:
        current_slice_slider.set_val(int(img3d.shape[0]/2))
        current_slice_slider.valmax = img3d.shape[0]-1
        current_slice_slider.ax.set_xlim(current_slice_slider.valmin, current_slice_slider.valmax)
    except NameError:
        current_slice_slider = Slider(slice_slider_axes, 'CT Slice', 0, img3d.shape[0]-1, valstep=1, valfmt='%0.0f')
        current_slice_slider.set_val(int(img3d.shape[0]/2))
        current_slice_slider.on_changed(on_slider_change)
    
    #plot the data, and connect
    im = ax.imshow(img3d[int(current_slice_slider.val), :, :], 
                         extent=[x_origin, x_origin + img3d.shape[1] * pixel_size, y_origin, y_origin + img3d.shape[2] * pixel_size],
                         cmap='Greys_r', vmin=hu_min, vmax=hu_max, animated = True, interpolation = 'nearest', origin = 'upper')
    contour_plot = ax.plot(contour_matrix[int(current_slice_slider.val)][0], contour_matrix[int(current_slice_slider.val)][1] )  
   
###############################################################################




if __name__ == '__main__':


    
    ###########################################################################
    # test inputs -- these will have to be edited to be put in a config file later
    ###########################################################################
    results_folder = 'turing_data' #directory to store dataframes/excel files of results
    dicom_data_folder = 'dicom_sample' #path to folder containing all folders that have ct data in them (ie in results folders are 0001, 0002, etc)
    human_rts_filename = 'Manual_only.dcm'
    machine_rts_filename = 'DL_Ottawa93.dcm'   
    structure_choices = ['Prostate', 'Rectum', 'Bladder']
    smooth_human_contours = True

   
    ###########################################################################
    # Initialize user info, and data available based on that
    ###########################################################################
    users = ['dgranville', 'mmacpherson', 'dlarussa']
    user = easygui.enterbox('Enter your TOH username')
    while user not in users:
        user = easygui.enterbox('Username not found.\nEnter your TOH username')
    # read in previous user data if it exists, if not create fresh set of user data
    if user + '.p' in os.listdir(results_folder):
        user_data = pickle.load( open( results_folder + '\\' + user + ".p", "rb" ) )
    else:
        user_data = []
        
    ###########################################################################
    # Allow user to select structure to evaluate
    ###########################################################################
    structure_wanted_human = easygui.choicebox('Select the structure to evaluate:', 'Structure Options', structure_choices)
    structure_wanted_machine = structure_wanted_human.upper() + '_1'    
    
    # generate list of patient folder names
    patient_list = [dicom_data_folder + '//' + x for x in os.listdir(dicom_data_folder)]
    
    # generate list of structure dicom paths (patient + machine/human combinations)
    structure_dicom_list = [x + '//' + machine_rts_filename for x in patient_list] + \
                           [x + '//' + human_rts_filename for x in patient_list]
    structure_list = []
    for dicom in structure_dicom_list:
        for structure in structure_choices:
            structure_list.append({'rts filename': dicom, 'structure': structure})
    

   # delete structure sets that have already been tests from this user!
    for data in user_data:
        if {'rts filename': data['rts filename'], 'structure':data['structure']} in structure_list:
            structure_list.remove({'rts filename': data['rts filename'], 'structure':data['structure']})
                

    
    ###########################################################################
    # Initialize plot and buttons
    ###########################################################################
    # Initialize plot features
    fig, ax = plt.subplots(1,1, figsize = [7.5,7.5], facecolor='black')
    plt.subplots_adjust(left=0.025)
    plt.subplots_adjust(right=0.975)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.15)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    slice_slider_axes = plt.axes([0.12, 0.02, 0.78, 0.03])
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Initialize buttons
    
    #machine button
    machine_button_axes = plt.axes([0.175, 0.075, 0.2, 0.05])
    machine_button = Button(machine_button_axes, 'MACHINE')
    machine_button.label.set_color('black')
    machine_button.label.set_fontweight('bold')
    machine_button.on_clicked(on_machine_button_click)
    
    #human button
    human_button_axes = plt.axes([0.625, 0.075, 0.2, 0.05])
    human_button = Button(human_button_axes, 'HUMAN')
    human_button.label.set_color('black')
    human_button.label.set_fontweight('bold')
    human_button.on_clicked(on_human_button_click)
    
    # w/l buttons
    wl_axes = plt.axes([0.12,0.74,0.2,0.2])
    wl_buttons = RadioButtons(wl_axes, ('Soft Tissue', 'Abdomen', 'Brain', 'Head', 'Lung'), activecolor = 'lightblue' )
    wl_buttons.on_clicked(wl_update)
    #change color of radio button edge
    for circle in wl_buttons.circles:
        circle.set_edgecolor('white')
    #make border around buttons black (ie hide it)
    for spine in wl_axes.spines.values():
        spine.set_edgecolor('black')
    #set initial w/l values (default to soft tissue)
    hu_min = 40 - 600/2
    hu_max = 40 + 600/2
            
    # Create the initial plot
    try:
        generate_plots()
    except UnboundLocalError:
        plt.close()
        exit()
    ###########################################################################
    
#    def handle_close(evt):
#        print('Closed Figure!')
#    fig.canvas.mpl_connect('close_event', handle_close)

# Smoothing examples
# https://stackoverflow.com/questions/30039260/how-to-draw-cubic-spline-in-matplotlib
# https://stackoverflow.com/questions/31464345/fitting-a-closed-curve-to-a-set-of-points
# example done in smoothing.py



