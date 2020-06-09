from connect import *
import numpy as np

case = get_current("Case")
exam = get_current('Examination')

def image_data(exam):
    '''takes the exam object as input and returns shaped numpy array
    of pixel data
    Array dimensions:
        1 = Slices
        2 = Columns
        3 = Rows
    '''
    dim_x = exam.Series[0].ImageStack.NrPixels.x
    dim_y = exam.Series[0].ImageStack.NrPixels.y
    dim_z = len(exam.Series[0].ImageStack.SlicePositions)

    rescale_intercept = exam.Series[0].ImageStack.ConversionParameters.RescaleIntercept
    rescale_slope = exam.Series[0].ImageStack.ConversionParameters.RescaleSlope
    pixel_representation = exam.Series[0].ImageStack.ConversionParameters.PixelRepresentation

    pixel_data = exam.Series[0].ImageStack.PixelData
    pixel_data = pixel_data.astype(np.float)
    length = len(pixel_data)

    evens = np.arange(0, length, 2, dtype=np.int)
    odds = np.arange(1, length, 2, dtype=np.int)

    if pixel_representation == 0:
        array = (pixel_data[evens]+pixel_data[odds]*256)
    else:
        array = (pixel_data[evens]+pixel_data[odds]*256)
        array = array.astype(np.int16)

    HU_data = np.reshape(array,(dim_z,dim_y,dim_x))

    HU_data = HU_data*rescale_slope+rescale_intercept

    return HU_data


for exam in case.Examinations:
    pixels = image_data(exam)
    #import matplotlib.pyplot as plt
    #plt.imshow(pixels[60],cmap = 'gray')
    #plt.show()
