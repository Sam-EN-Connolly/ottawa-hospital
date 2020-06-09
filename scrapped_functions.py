'''
Functions used in development that have been abandoned 
'''

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