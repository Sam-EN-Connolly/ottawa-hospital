# ottawa-hospital
 Organ deformation modeling for the ottawa hospital summer student project

This collection of code was written for a summer project at the Ottawa Hospital Medical Phsyics Department. 
The goal of this project is to characterize and model the deformation of the liver and prostate 
in cancer patients treated on the CyberKnife system. 

Functions in the package are oriented around the use of the pydicom package for reading DICOM files, 
as well as reading in data through the software Raysation. 

This package includes functions for : 

File I/O : 
- Read extranted patient treatment files to find fiducial locations, as XML files
- Writting and reading CSV files 

Processing DICOM files : 
- Opening and reading DICOM files
- Converting CT images to ndarrays, and cenverting coordinate systems
- Finding fiducials in images through thresholding

Processing DICOM structure files : 
- Opeing and reading structure files 
- Converitng contours to dictionaries and ndarrays
- Calculating mean surface distance, residual mean square error, hausdorff distance, and dice coefficient between two contours
- Performing isometric scaling of contours
- Determining ROI center of mass

Determining fiducial spacial properties
- Finding changes in fiducial locations between images
- Finding fiducial center of mass
- Finding avaerage distance to fiducial or ROI center of mass

Interfacing with Raystation 7 : 
- Extracting image data and converting to ndarray 
- Finding location of, creating, and moving POIs
- Determining fiducial location through POI locations and thresholding
Note about Raystation 7 : 
The code written for Raystation 7 is in CPython 2.7. Raystation 7 supports both 
IronPython 2.7 and CPython 2.7. However, IronPython showed signifigant memory limitations.

Acronyms : 
- ROI : region of interest
- POI : point of interst
- TPCT : treatment planning CT


Data structures : 
The main data structures used in the overall anaysis, are dictionsaries. These distionaries hold the metadata, fiducial data, contour data, and contour comparison data. Depending on the anysis performed, the structure of these dictionaries will be slightly different. 

metatdata_dict : dictionary 
- First level of dictionary keys : MRN
- Second level of dictionary keys image type : either TPCT or FUSED
- Third level of dictionary hold the variables with keys : 'image shape', 'origin', 'pixel spacing' 

fiducial_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}, 'CHANGE' : {...}}}
ROI_contour_dict = {} # {MRN : {'TPCT' : {...}, 'FUSED' : {...}, 'FUSED_ISO' : {...}}}
contour_comparison_dict = {} # {MRN : {'TPCT_to_FUSED' : {...}, 'TPCT_to_FUSED_ISO' : {...}}}